# Code for paper: Semantic-Preserving Surgical Video Retrieval with Phase and Behavior Coordinated Hashing

This repository contains the code implementation of the Semantic-Preserving Surgical Video Retrieval (SPSVR) framework.

## Requirements

- numpy >= 1.23.1
- torch >= 1.11.0
- torchvision >= 0.12.0
- sacred >= 0.8.2

## Preprocessing datasets

To start using the code repository for training the SPSVR model and generating hash codes of surgical videos, the original datasets need to be processed as follows.

### Extract frames

We need to decompose the video into frames. After extracting frames, the directory structure should be as follows.

```text
dataset_root
├─ data
│  ├─ video01
│  │    ├─ 00000000.jpg
│  │    ├─ 00000001.jpg
│  │    ├─ 00000002.jpg
│  │    ├─ ...
│  ├─ video02
│  ├─ video03
│  └─ ...
└─ label
```

### Generate CSV files for labels

Training the model requires surgical phase labels, we need to write the phase labels of each video into a csv file, as follows.

```text
video01/00000000.jpg,0
video01/00000001.jpg,0
video01/00000002.jpg,1
```

The directory structure should be as follows.

```text
dataset_root
├─ data
└─ label
   ├─ video01.csv
   ├─ video02.csv
   ├─ video03.csv
   └─ ...
```

## Training

```bash
python train.py --cfg config/<dataset>.yaml
```

You can modify the configurations in `config/<dataset>.yaml` based on your own environment.

## Generate hash codes

```bash
python generate.py --cfg config/<dataset>.yaml
```

You can modify the `model_path` entry in `config/<dataset>.yaml` to load the pre-trained model weights.
