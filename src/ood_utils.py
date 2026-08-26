import torch
import random
import numpy as np
import os
import src.ood_calculate_log as callog
from transformers import DataCollatorForLanguageModeling, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    batch = data_collator(batch)
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
    input_mask = torch.tensor(batch["attention_mask"], dtype=torch.float)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
    }
    return outputs

def detection_performance(scores, Y, outf, tag='TMP'):
    """
    Measure the detection performance
    return: detection metrics
    """
    os.makedirs(outf, exist_ok=True)
    num_samples = scores.shape[0]
    l1 = open('%s/confidence_%s_In.txt'%(outf, tag), 'w')
    l2 = open('%s/confidence_%s_Out.txt'%(outf, tag), 'w')
    y_pred = scores

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{},".format(-y_pred[i]))
        else:
            l2.write("{},".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, [tag])
    return results