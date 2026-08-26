import datasets
import random
import json

datasets.logging.set_verbosity(datasets.logging.ERROR)


def load(task_name, tokenizer, max_seq_length=512, is_id=False, base_unlearn_path=None, base_ood_path=None):

    sentence1_key = "function"
    if "codellama" in task_name and "ood" not in task_name:
        data = load_codellama_topic(task_name.split("_")[-1], base_unlearn_path)
    elif "ood_codellama" in task_name:
        data = load_codellama_notopic(base_ood_path)

    def preprocess_function(examples):
        inputs = (examples[sentence1_key],)
        result = tokenizer(*inputs, padding='max_length', max_length=max_seq_length, truncation=True)
        return result

    train_dataset = list(map(preprocess_function, data['train'])) if 'train' in data and is_id else None
    test_dataset = list(map(preprocess_function, data['test'])) if 'test' in data else None
    return train_dataset, test_dataset

def load_codellama_topic(topic, path):
    print("***", path)
    with open(path, encoding='utf-8') as fp:
        train_data = json.load(fp)

    forget_ratio = 0.8
    length = forget_ratio * len(train_data)

    forget_index_list = random.sample(range(len(train_data)), int(length))
    retain_index_list = list(set(range(len(train_data))) - set(forget_index_list))

    train_dataset = [train_data[i] for i in forget_index_list]
    test_dataset = [train_data[i] for i in retain_index_list]

    return {'train': train_dataset, 'test': test_dataset}

def load_codellama_notopic(path):
    with open(path, encoding='utf-8') as fp:
        data = json.load(fp)
    # Use "retain" field as the function text for OOD (retain) data
    train_dataset = [{"function": entry["retain"]} for entry in data if entry.get("retain")]
    return {'train': train_dataset, 'test': train_dataset}
