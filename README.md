# MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging
![The overall architecture of MIRNet.](figs/MIRNet.png)
The overall architecture of MIRNet. The central diagram shows the main workflow: a pretrained MAE extracts image embeddings, a label graph is built from statistical dependencies, and a domain-aware GAT captures higher-order label correlations. The model is then trained via a constraint-aware optimization mechanism. The left panel details domain-aware pretraining and the GAT, while the right panel illustrates label graph construction and constraint-aware optimization.

# Dataset: TongueAtlas-4K
TongueAtlas-4K is a high-resolution dataset consisting of 4,000 tongue images collected from clinical patients, annotated by certified Traditional Chinese Medicine (TCM) experts.
The dataset is available [here](https://doi.org/10.5281/zenodo.17557646).

# Installation
Before running the project, make sure to install all required dependencies:
```
pip install -r requirements.txt
```

# Pre-training
```
python pretrain.py \
    --batch_size 512 \
    --epochs 500 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --data_path path/to/your_datasets \
    --blr 5e-4 \
    --output_dir path/to/save_results
```

# Fine-tuning
```
python finetune.py \
    --batch_size 200 \
    --epochs 200 \
    --model vit_base_patch16 \
    --nb_classes 22 \
    --input_size 224 \
    --blr 1e-3 \
    --finetune path/to/your_pretrained_model \
    --train_data_dir path/to/your/train_datasets \
    --valid_data_dir path/to/your/validation_datasets \
    --train_label_file path/to/your/train_label_file \
    --valid_label_file path/to/your/validation_label_file \
    --label_graph_file path/to/your_label_graph_file
```

# Evaluation
```
python eval.py \
    --batch_size 200 \
    --nb_classes 22 \
    --input_size 224 \
    --data_dir path/to/your_data_dir \
    --label_file path/to/your_label_file \
    --model_path path/to/your_model.pth \
    --label_graph_file path/to/your_label_graph_file \
    --result_path path/to/save_results \
    --output_file your_output_file_name
```

# Paper
Our paper has been accepted by AAAI 2026 (to appear).
