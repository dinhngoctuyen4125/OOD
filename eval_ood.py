"""
Evaluate OOD Detector on a dataset (e.g. D_forget).
Loads trained OOD checkpoint, computes OOD weights per sample,
and prints a summary of detection results.
"""
import os
import json
import argparse
import math
import pickle

import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from scipy.stats import norm

from src.ood_model_selector import RobertaForSelector_inference


# ---- GMM weighting functions ----

def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    return np.sum([w * norm.cdf(x, m, s) for w, m, s in zip(weights, means, stds)])


def obtain_weights(input_x, gmm, x0):
    cp_x = gmm_cdf(input_x, gmm)
    cp_sym = gmm_cdf(2 * x0 - input_x, gmm)

    cp_sum = 1 - max(cp_x, cp_sym) + min(cp_x, cp_sym)
    cp_sum *= 10  # scaling_factor
    range_th = 2

    w = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    if w > 0.9:
        w = 1.2
    elif 0.3 < w <= 0.4:
        w = w
    else:
        w = 0

    return w


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD Detector")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON data file (e.g. D_forget.json)")
    parser.add_argument("--text_field", type=str, default="function",
                        help="JSON field name containing the code text to evaluate")
    parser.add_argument("--ood_weights", type=str, required=True,
                        help="Path to OOD checkpoint directory")
    parser.add_argument("--ood_base_model", type=str, default="microsoft/codebert-base",
                        help="Base model for OOD detector")
    parser.add_argument("--ood_setting_name", type=str, default="codellama",
                        help="OOD setting name (used to construct checkpoint filenames)")
    parser.add_argument("--ood_type", type=str, default="_all",
                        help="OOD type suffix (e.g. '_all', '_torch')")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for OOD scoring")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for tokenizer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    print(f"[*] Loading data from: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"    Total samples: {len(data)}")

    # ---- Build OOD checkpoint paths ----
    ood_types = [x for x in args.ood_type.split("_") if x]
    t = ood_types[0] if ood_types else "all"
    wp = os.path.join(args.ood_weights,
                      f"{args.ood_setting_name}_{t}_ood_{args.ood_setting_name}")

    roberta_path = wp + "_roberta_ocsvm"
    ocsvm_path = wp + "_ocsvm.pkl"
    gmm_w_path = wp + "_gmm_w_ocsvm.pkl"
    threshold_path = wp + "_threshold_ocsvm.json"
    mean_list_path = wp + "_mean_list_ocsvm.pt"
    precision_list_path = wp + "_precision_list_ocsvm.pt"
    fea_list_path = wp + "_fea_list_ocsvm.pt"

    print(f"[*] Loading OOD checkpoint from: {args.ood_weights}")
    print(f"    Prefix: {wp}")

    # ---- Load OOD components ----
    ood_tokenizer = RobertaTokenizer.from_pretrained(args.ood_base_model)

    ood_model = RobertaForSelector_inference(
        args.ood_base_model, lora_path=roberta_path, projection_dim=100
    ).to(device)
    ood_model.eval()

    with open(ocsvm_path, "rb") as f:
        ocsvm = pickle.load(f)
    with open(gmm_w_path, "rb") as f:
        gmm_w = pickle.load(f)
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
    x0 = threshold_data[0]
    threshold = threshold_data[1]
    train_acc = threshold_data[2]

    mean_list = torch.load(mean_list_path, map_location=torch.device(device))
    precision_list = torch.load(precision_list_path, map_location=torch.device(device))
    fea_list = torch.load(fea_list_path, map_location=torch.device(device))

    print(f"    OCSVM threshold: {threshold:.6f}")
    print(f"    GMM x0: {x0:.6f}")
    print(f"    Training accuracy: {train_acc:.4f}")
    print(f"[*] OOD detector loaded successfully.")

    # ---- Evaluate ----
    all_weights = []
    all_ocsvm_scores = []
    weight_distribution = {}

    print(f"\n[*] Running OOD evaluation (batch_size={args.batch_size})...")
    for start_idx in tqdm(range(0, len(data), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(data))
        batch = data[start_idx:end_idx]

        # Extract text for OOD scoring
        texts = []
        for sample in batch:
            text = sample.get(args.text_field, "")
            if not text:
                # Fallback: try common field names
                for fallback_key in ["probing input new", "probing input", "code", "text"]:
                    text = sample.get(fallback_key, "")
                    if text:
                        break
            texts.append(text if text else "")

        # Tokenize
        ood_input = ood_tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=args.max_seq_length, return_tensors="pt"
        )

        # Compute Mahalanobis scores
        with torch.no_grad():
            mah_scores = ood_model.get_unsup_Mah_score_s(
                ood_input, mean_list, precision_list, fea_list
            )[:, 1:]

        # OCSVM scoring
        ocsvm_scores = ocsvm.score_samples(mah_scores)

        # Compute weights
        for score in ocsvm_scores:
            w = obtain_weights(score, gmm_w, x0)
            all_weights.append(w)
            all_ocsvm_scores.append(score)

            key = str(round(w, 3))
            weight_distribution[key] = weight_distribution.get(key, 0) + 1

    # ---- Summary ----
    all_weights = np.array(all_weights)
    all_ocsvm_scores = np.array(all_ocsvm_scores)
    n = len(all_weights)

    n_zero = int(np.sum(all_weights == 0))
    n_mid = int(np.sum((all_weights > 0) & (all_weights < 1.0)))
    n_high = int(np.sum(all_weights >= 1.0))

    # OCSVM binary classification
    n_id = int(np.sum(all_ocsvm_scores > threshold))
    n_ood = int(np.sum(all_ocsvm_scores <= threshold))

    print("\n" + "=" * 65)
    print("OOD EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Data file:               {args.data_path}")
    print(f"  Text field:              {args.text_field}")
    print(f"  OOD checkpoint:          {args.ood_weights}")
    print(f"  OCSVM threshold:         {threshold:.6f}")
    print(f"  GMM x0:                  {x0:.6f}")
    print()
    print(f"  Total samples:           {n}")
    print()
    print("  --- OCSVM Binary Classification ---")
    print(f"  In-Distribution (ID):    {n_id} ({n_id / n * 100:.1f}%)")
    print(f"  Out-of-Distribution:     {n_ood} ({n_ood / n * 100:.1f}%)")
    print()
    print(f"  --- OCSVM Score Statistics ---")
    print(f"  Mean score:              {np.mean(all_ocsvm_scores):.6f}")
    print(f"  Std score:               {np.std(all_ocsvm_scores):.6f}")
    print(f"  Min / Max:               {np.min(all_ocsvm_scores):.6f} / {np.max(all_ocsvm_scores):.6f}")
    print()
    print("  --- Soft Weight Distribution (GMM → LoRA Activation) ---")
    print(f"  Mean weight:             {np.mean(all_weights):.4f}")
    print(f"  Min / Max:               {np.min(all_weights):.4f} / {np.max(all_weights):.4f}")
    print(f"  w = 0   (LoRA OFF):      {n_zero} ({n_zero / n * 100:.1f}%)")
    print(f"  0 < w < 1 (partial):     {n_mid} ({n_mid / n * 100:.1f}%)")
    print(f"  w >= 1  (LoRA FULL):     {n_high} ({n_high / n * 100:.1f}%)")
    print(f"  LoRA activation rate:    {(n - n_zero) / n * 100:.1f}%")
    print()
    print(f"  Weight value counts:")
    for k in sorted(weight_distribution.keys(), key=lambda x: float(x)):
        count = weight_distribution[k]
        print(f"    w={k:>6s}: {count:>5d} ({count / n * 100:5.1f}%)")
    print("=" * 65)

    # ---- Save results ----
    result_file = args.data_path.replace(".json", "_ood_eval.json")
    result_data = {
        "data_path": args.data_path,
        "ood_checkpoint": args.ood_weights,
        "total_samples": n,
        "ocsvm_threshold": threshold,
        "gmm_x0": x0,
        "ocsvm_classification": {
            "in_distribution": n_id,
            "out_of_distribution": n_ood,
        },
        "ocsvm_score_stats": {
            "mean": float(np.mean(all_ocsvm_scores)),
            "std": float(np.std(all_ocsvm_scores)),
            "min": float(np.min(all_ocsvm_scores)),
            "max": float(np.max(all_ocsvm_scores)),
        },
        "soft_weight_summary": {
            "mean": float(np.mean(all_weights)),
            "min": float(np.min(all_weights)),
            "max": float(np.max(all_weights)),
            "lora_off_w0": n_zero,
            "lora_partial": n_mid,
            "lora_full": n_high,
            "lora_activation_rate": round((n - n_zero) / n * 100, 2),
            "weight_distribution": weight_distribution,
        },
    }
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n[*] Results saved to: {result_file}")


if __name__ == "__main__":
    main()
