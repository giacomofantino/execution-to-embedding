# From Execution to Embedding: Enriching Code Representations with Data Difference Signals for Comment Generation

This repository supports the reproducibility of the study "From Execution to Embedding: Enriching Code Representations with Data Difference Signals for Comment Generation", providing scripts, datasets, and model code for training comment-generation models from data wrangling notebooks.

All the data, checkpoints and notebooks are available from [zenodo](https://zenodo.org/records/17198749?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM4NTFiZDg3LTA4NWYtNDlkYi04NjI1LTUyMGY0MjY5MDZjNiIsImRhdGEiOnt9LCJyYW5kb20iOiJmYjQ4MDY1NDczN2IzYmNjYjM5NGFmMjAzOTg4MDY1YiJ9.9MhKjGBB_B6sLUdjyumFNDSnyJphSyZsYn6uCnaed3vcSp5luDlm2Q1nDkfHvUGNwXHBu_eqo9c-UaxEzXNmyw).

The ZIP file contains three main directories related to dataset creation and model training:

## 📁 build
Contains notebooks, input data, and scripts/libraries used to extract samples and build the dataset.

#### Structure:

```
build/
├── install_libs.py
├── notebooks/
│ └── tracker.py
│ └── ...
└── inputs/
└── libs.txt
```

Inside notebooks there are the retrieve notebooks from Kaggle, while inside inputs/ the datasets used by the notebooks for data analysis.
This file will extract both semantic difference and also include examples of data before and after the execution:
`<ADDED> whole_voters <VAL> 961 <ADDED> weighted_voters <VAL> 1376`

Later for the model we decided to remove this part regarding the value and keep only the semantic difference, thus the data the model will see is:
`<ADDED> whole_voters <ADDED> weighted_voters`

Future work will also include the possibility of understanding how to include this data in the embedding to further enhance the data encoder embeddings.

## 📁 dataset
Final dataset, divided into training, validation, and test subsets.  
The test set is further split into two: `diff` and `no_diff`.

### structure:
```
dataset/my_data/
├── train_data.jsonl
├── val_data.jsonl
├── test_data_diff.jsonl
└── test_data_no_diff.jsonl
```

These files must be place inside the `dataset` directory. Note that for pre-train the code encoder you have to create a `data` directory and use the CodeXGlue dataset, specifically the `python` subset.

## 📁 model

Contains the checkpoint for the model analyzed in the paper, to replicate the experiments. Must be placed inside the `model` directory.

# Co-Attention
All files necessary to train and test the model, assuming dataset files.

#### Contents:
```
dataset/                  # folder with the dataset
co-attention/
├── model/                # folder with model checkpoints
├── components.py         # basic components of the model
├── finetune_code.py      # Main classes for finetuning the unimodal model
├── libs.txt              # The libraries for reproducing the environments
├── model_ft.py           # Main classes for finetuning the multimodal model (contains Co-Attention)
├── model_pt.py           # Main classes for pretraining the model on CodeXGlue
├── pretrain_diff.py      # Main classes for pretraining diff encoder on the dataset
├── pretrain_encoders.py  # Main classes for jointly pretraining encoders on the dataset
└── train_model.py        # Main file for training models
```

The main file is train_model.py, which contains the main logic to retrieve checkpoints, train and test models.

The parameters are:
```
--mode              # Type of operation/model:
                    # 'pretrain_code'      → Pretrain on code only
                    # 'pretrain_diff'      → Pretrain on diff only
                    # 'pretrain_encoders'  → Jointly pretrain both encoders
                    # 'finetune'           → Fine-tune multimodal model (code + diff)
                    # 'finetune_code'      → Fine-tune code-only model

--task              # 'train' or 'test'

--retrieve_pretrain # (int) Use checkpoint from a pretraining epoch for finetuning

--epoch             # (str) Specify which model epoch to load for testing

--diff_epoch        # (str) Load specific epoch of diff-pretrained model

--encoders_epoch    # (str) Use pretrained encoders checkpoint from this epoch

--diff_epoch        # (str) Use pretrained data-diff encoder checkpoint from this epoch

--phase             # (int) which phase of train encoders to use
```
## 📝 How to Reproduce

1. **Dataset building**:
   - To extract the dataset, first install dependencies from libs.txt and then run tracker.py inside the notebooks/ directory.
   - Navigate to `build/`
   - Run `pip install -r libs.txt` to install the required libraries.
   - Navigate to `notebooks/`
   - start the tracker script which execute notebooks and tracks the data by running `python tracker.py`
   - after the execution the datasets will be inside the `build/` folder.

2. **Model training/testing**:
   - Navigate to `train_model/`
   - Run `pip install -r libs.txt` to install the required libraries.
   - Based on which type of model we want to train or test there are various calls. Here some examples:
   - Test a fine-tuned multimodal model:

   ```python train_model.py --task=test --mode=finetune --epoch=4_all```
   - Test a code-only fine-tuned model:
   
   ```python train_model.py --task=test --mode=finetune_code --epoch=5```
   - Fine-tune a code-only model from a code pretraining checkpoint:
   
   ```python train_model.py --mode=finetune_code --task=train --retrieve_pretrain=5```
   - Fine-tune a multimodal model using pretrained encoder weights:
   
   ```python train_model.py --mode=finetune --task=train --retrieve_pretrain=5 --encoders_epoch=10_all```
   - Pretrain encoders jointly (Code + Diff):
   
   ```python train_model.py --mode=pretrain_encoders --task=train --retrieve_pretrain=5 --diff_epoch=2_A1```

3. **Requirements**:
   - This project was developed and tested under the following environment:
   - OS: Pop!_OS 22.04 LTS
   - CPU: Intel(R) Core(TM) i7-2600 @ 3.40GHz
   - RAM: 16 GiB
   - GPU: NVIDIA GeForce RTX 2070 (8 GiB VRAM)
   - Python: 3.10.12
   - PyTorch: 2.6.0 (with CUDA 12.7 support)

   To ensure full compatibility, use the provided libs.txt to recreate the runtime.

## 📄 License
This project is licensed under the Creative Commons Attribution 4.0 International License.