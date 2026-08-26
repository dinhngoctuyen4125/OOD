# OOD Detector Training

## Setup

```bash
conda create -n ood python=3.10
conda activate ood
pip install -r requirements.txt
```

## Train OOD Detector

```bash
sudo nohup bash train_ood.sh > logs/train_ood.log 2>&1 &
```