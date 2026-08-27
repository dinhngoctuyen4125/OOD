import torch
import random
import numpy as np
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