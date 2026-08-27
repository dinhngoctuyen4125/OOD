
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import RobertaModel
import sklearn.covariance
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig, get_peft_model


def entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy



class RobertaForSelector(nn.Module):
    def __init__(self, model_name, projection_dim):
        super().__init__()
        peft_config = LoraConfig(task_type="FEATURE_EXTRACTION",
                                 r=8,  # Rank Number
                                 lora_alpha=32,  # Alpha (Scaling Factor)
                                 lora_dropout=0.1,  # Dropout Prob for Lora
                                 target_modules=["query", "key", "value"],
                                 # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
                                 bias='none', )

        roberta = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
        peft_model = get_peft_model(roberta, peft_config)
        print('PEFT Model')
        peft_model.print_trainable_parameters()
        self.roberta = peft_model

        self.roberta_k = RobertaModel.from_pretrained(model_name, output_hidden_states=True)

        for param_k in self.roberta_k.parameters():
            param_k.requires_grad = False


    def forward(self, batch_mlm=None, batch=None, steps=0, dataloader=None):
        outputs = self.roberta(
            input_ids=batch_mlm["input_ids"],
            attention_mask=batch_mlm["attention_mask"],
        )

        info_loss = 0

        with torch.no_grad():

            outputs_k = self.roberta_k(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        for i in range(13):
            z_1 = torch.mean(outputs.hidden_states[i], dim=1, keepdim=False)
            z_2 = torch.mean(outputs_k.hidden_states[i], dim=1, keepdim=False)
            sim_mat = torch.einsum('nc,ck->nk', [z_1, z_2.T.detach()])
            s_dist = F.softmax(sim_mat, dim=1)
            info_loss += torch.mean(entropy(s_dist))

        return torch.tensor(0.0, device=batch_mlm["input_ids"].device), info_loss

    def sample_X_estimator(self, dataloader):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                all_layer_features[i].append(layer_mean_fea.data.cpu())

        mean_list = []
        precision_list = []
        fea_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1))
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.cuda())
            precision_list.append(temp_precision.cuda())

        return mean_list, precision_list, fea_list

    def get_unsup_Mah_score(self, dataloader, sample_mean, precision, fea_list):
        total_mah_scores = []
        num_layers = 13
        for i in range(num_layers):
            total_mah_scores.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            batch_all_features = []
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                batch_all_features.append(layer_mean_fea.data)

            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean
                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                out_feas = F.normalize(out_features, dim=-1)
                cs_score = out_feas @ fea_list[i].t().cuda()
                cs_score = torch.max(cs_score, dim=1)[0]
                all_score = -cs_score * 1000. + gaussian_score
                total_mah_scores[i].extend(all_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)

        return np.concatenate(total_mah_scores, axis=1)


class RobertaForSelector_inference(nn.Module):
    def __init__(self, model_name, lora_path, projection_dim):
        super().__init__()

        roberta = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
        peft_model = PeftModel.from_pretrained(
            roberta,
            lora_path,
        )
        self.roberta = peft_model

    def sample_X_estimator(self, dataloader):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])


        for step, batch in enumerate(tqdm(dataloader)):
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                all_layer_features[i].append(layer_mean_fea.data.cpu())

        mean_list = []
        precision_list = []
        fea_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1))
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.cuda())
            precision_list.append(temp_precision.cuda())

        return mean_list, precision_list, fea_list

    def get_unsup_Mah_score_s(self, ood_input, sample_mean, precision, fea_list):
        total_mah_scores = []
        num_layers = 13
        for i in range(num_layers):
            total_mah_scores.append([])


        batch_all_features = []
        self.eval()
        outputs = self.roberta(
            input_ids=ood_input['input_ids'].cuda(),
            attention_mask=ood_input['attention_mask'].cuda(),
        )
        all_hidden_feats = outputs.hidden_states

        for i in range(num_layers):
            layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
            batch_all_features.append(layer_mean_fea.data)

        for i in range(len(batch_all_features)):
            batch_sample_mean = sample_mean[i]
            out_features = batch_all_features[i]
            zero_f = out_features - batch_sample_mean
            gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
            out_feas = F.normalize(out_features, dim=-1)
            cs_score = out_feas @ fea_list[i].t().cuda()
            cs_score = torch.max(cs_score, dim=1)[0]
            all_score = -cs_score * 1000. + gaussian_score
            total_mah_scores[i].extend(all_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)

        return np.concatenate(total_mah_scores, axis=1)
