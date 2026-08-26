import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from src.ood_utils import set_seed, collate_fn, detection_performance

from sklearn import svm
from sklearn.mixture import GaussianMixture as GMM

import os
from src.ood_model_selector import RobertaForSelector
import json

import warnings
from src.ood_data import load
import pickle
import math

from scipy.stats import norm


warnings.filterwarnings("ignore")
torch.set_num_threads(10)


def train(args, model, train_dataset, test_dataset, benchmarks, save_name):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    acc_g = 0

    def detect_ood(acc_global, seed):
        seed = str(seed)
        mean_list, precision_list, fea_list = model.sample_X_estimator(train_dataloader)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        test_mah_vanlia = model.get_unsup_Mah_score(test_dataloader, mean_list, precision_list, fea_list)[:, 1:]
        train_mah_vanlia = model.get_unsup_Mah_score(train_dataloader, mean_list, precision_list, fea_list)[:, 1:]

        for _, ood_dataset in benchmarks:
            ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
            ood_mah_vanlia = model.get_unsup_Mah_score(ood_dataloader, mean_list, precision_list, fea_list)[:, 1:]

            ood_labels = np.ones(shape=(ood_mah_vanlia.shape[0],))
            test_labels = np.zeros(shape=(test_mah_vanlia.shape[0],))

            test_mah_scores = test_mah_vanlia
            ood_mah_scores = ood_mah_vanlia
            train_mah_scores = train_mah_vanlia

            np.random.shuffle(test_mah_scores)
            np.random.shuffle(ood_mah_scores)

            if args.ood == 'ocsvm':
                c_lr = svm.OneClassSVM(nu=0.1, kernel='linear', degree=2)
                c_lr.fit(train_mah_scores)

                test_scores = c_lr.score_samples(test_mah_scores)
                ood_scores = c_lr.score_samples(ood_mah_scores)
                train_scores = c_lr.score_samples(train_mah_scores)
                Y_test = np.concatenate((ood_labels, test_labels))

                scores = np.concatenate((ood_scores, test_scores))
                results = detection_performance(scores, Y_test, 'mah_logs', tag='TMP')
                neg_resuls = detection_performance(-scores, Y_test, 'feats_logs', tag='TMP')

                if sum(results["TMP"].values()) < sum(neg_resuls["TMP"].values()):
                    results = neg_resuls
                auroc = results['TMP']['AUROC']
                print(args.ood + ' current auroc: {:.3f}'.format(auroc))

                ## test the new weighting function
                gmm_w, x0 = weighting_func_gmm(train_scores, test_scores)
                w_res_in_train = [obtain_weights(x, gmm_w, x0) for x in train_scores]
                w_res_in_test = [obtain_weights(x, gmm_w, x0) for x in test_scores]
                w_res_ood = [obtain_weights(x, gmm_w, x0) for x in ood_scores]

                print(np.mean(np.array(w_res_in_train)), np.mean(np.array(w_res_in_test)), np.mean(np.array(w_res_ood)))

                threshold = np.max(train_scores)  # 99% of the training set as the threshold

                test_labels_prediction = (test_scores <= threshold).astype(int)
                ood_labels_prediction = (ood_scores <= threshold).astype(int)
                Y_predict = np.concatenate((ood_labels_prediction, test_labels_prediction))
                acc = (Y_predict == Y_test).mean()
                print('Test set accuracy: {:.3f}'.format(acc))

                if acc > acc_global:
                    ood_path = f"./ood_checkpoints_{save_name}_{seed}"

                    if not os.path.exists(ood_path):
                        os.mkdir(ood_path)

                    torch.save(mean_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_mean_list_ocsvm.pt")
                    torch.save(precision_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_precision_list_ocsvm.pt")
                    torch.save(fea_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_fea_list_ocsvm.pt")

                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_gmm_w_ocsvm.pkl", "wb") as output_file:
                        pickle.dump(gmm_w, output_file)
                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_ocsvm.pkl", "wb") as output_file:
                        pickle.dump(c_lr, output_file)
                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_threshold_ocsvm.json", 'w') as f:
                        json.dump([x0, threshold, acc], f)
                    print("SAVE", "CURRENT BEST ACC: ", acc)
                    acc_global = acc
                    model.roberta.save_pretrained(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_roberta_ocsvm")

                return acc_global



    num_steps = 0
    acc_g = detect_ood(acc_g, args.seed)
    for epoch in range(int(args.num_train_epochs)):
        print("start training")
        model.zero_grad()
        for batch in tqdm(train_dataloader):
            model.train()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(batch, batch, num_steps, train_dataloader)
            _, moco_loss = outputs
            loss = moco_loss
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            print('Step:', num_steps, 'moco_loss: ', moco_loss.item())
        acc_g = detect_ood(acc_g, args.seed)
        print("Epoch Accuracy: ", acc_g)


def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    x0 = (mean1 + mean2) / 2

    return gmm, x0

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# 6. the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2 # 0.5

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    return w_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=2.0, type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ood", type=str, default="ocsvm")

    parser.add_argument("--unlearn_dataset", default="codellama_torch", type=str)
    parser.add_argument("--ood_dataset", type=str, default="ood_codellama")
    parser.add_argument("--base_unlearn_path", type=str, default="./data/codellama/SD/")
    parser.add_argument("--base_ood_path", type=str, default="./data/codellama/RD/not_torch.json")
    parser.add_argument("--save_name", type=str, default="codellama")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaForSelector(args.model_name_or_path, projection_dim=100)
    model.to(args.device)
    datasets = [args.unlearn_dataset, args.ood_dataset]
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.unlearn_dataset:
            train_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True, base_unlearn_path=args.base_unlearn_path, base_ood_path=args.base_ood_path)
        elif dataset == args.ood_dataset:
            _, original_val_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True, base_unlearn_path=args.base_unlearn_path, base_ood_path=args.base_ood_path)
            benchmarks = ((dataset, original_val_dataset),) + benchmarks
    train(args, model, train_dataset, test_dataset, benchmarks, save_name=args.save_name)


if __name__ == "__main__":
    main()
