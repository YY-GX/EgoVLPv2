# EgoVLPv2 for Parkinson Action Recognition

This guide provides instructions for using EgoVLPv2 to fine-tune and evaluate models on Parkinson action recognition datasets.

## Installation

For installation instructions and general setup, please refer to the main [README.md](README.md) file.

**Prerequisites:**
- Python 3.8+
- PyTorch
- CUDA-compatible GPU(s)
- Conda environment (recommended: `egovlpv2`)

## Quick Start Commands

### 1. Data Preparation

```bash
python utils/prepare_annotations.py \
    --annotation_dir /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/annotations \
    --clip_duration 2 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --video_dir data \
    --output_dir /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/annotations
```

**What it does:** Processes raw video annotations and creates train/validation/test splits for training. This script:
- Takes raw annotation CSV files with timestamps and action labels
- Splits videos into clips of specified duration (2 seconds)
- Creates train (70%), validation (15%), and test (15%) splits
- Supports two split modes: random clip-based or video-based splitting
- Outputs CSV files ready for training data loading

### 2. Fine-tuning Training

```bash
PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2 python multinode_train_epic.py \
    --save_dir finetune_parkinson \
    --config ./configs/ft/parkinson_ego.json \
    --print_freq 100
```

**What it does:** Main training script for fine-tuning EgoVLPv2 on your Parkinson dataset. This script:
- Loads the pre-trained EgoVLPv2 model from `checkpoints/EgoVLPv2.pth`
- Fine-tunes on Parkinson action classification task
- Uses configuration from `parkinson_ego.json` (16-frame video input, RoBERTa text encoder)
- Saves checkpoints in timestamped subfolders under `finetune_parkinson/`
- Supports multi-GPU training (configured for 8 GPUs)
- Uses AdamW optimizer with learning rate 0.0001 and 50 training epochs

### 3. Training Monitoring with TensorBoard

```bash
cd /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/finetune_parkinson
tensorboard --logdir=tf --port=6006 --host=0.0.0.0
ssh -L 6006:localhost:6006 yygx@oprime.cs.unc.edu
# Then open http://localhost:6006 in your browser
```

**What it does:** Sets up TensorBoard to monitor training progress:
- Starts TensorBoard server on port 6006
- Creates SSH tunnel to access TensorBoard from your local machine
- View training metrics, loss curves, and model performance in real-time

### 4. Model Evaluation (Specific Checkpoint)

```bash
python eval_action_classification_test_valstyle.py \
    -c ./configs/ft/parkinson_ego.json \
    --checkpoint ./finetune_parkinson/20250702_172406/checkpoint-epoch80.pth \
    --save_dir ./finetune_parkinson/20250702_172406 \
    --batch_size 8
```

**What it does:** Evaluates a specific checkpoint on the test set using validation-style inference:
- Loads a trained model checkpoint (e.g., epoch 80)
- Runs inference on the test dataset
- Computes accuracy metrics
- Can save predictions to CSV for detailed analysis

### 5. Best Checkpoint Evaluation

```bash
python eval_best_ckpt.py \
    --config ./configs/ft/parkinson_ego.json \
    --model_ckpt /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/finetune_parkinson/20250709_013137/model_best.pth \
    --output_dir /mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/finetune_parkinson/20250709_013137/best_val_ckpt \
    --device cuda
```

**What it does:** Comprehensive evaluation of the best validation checkpoint:
- Loads the best model based on validation performance
- Runs evaluation on the test set
- Generates confusion matrix and detailed metrics
- Identifies misclassified examples for analysis
- Saves comprehensive evaluation results

## Typical Workflow

1. **Prepare your data** using `prepare_annotations.py`
2. **Start training** using `multinode_train_epic.py`
3. **Monitor progress** using TensorBoard
4. **Evaluate performance** using either evaluation script
5. **Analyze results** from the saved outputs

## Configuration

The main configuration file `./configs/ft/parkinson_ego.json` contains:
- Model architecture settings (16-frame video, 224x224 resolution)
- Training hyperparameters (learning rate, batch size, epochs)
- Data loading settings (dataset paths, transforms)
- Loss function configuration

## Quick Performance Check

**Fastest way to check model performance:** Use TensorBoard during training to monitor the validation loss curve. This gives you the most accurate and real-time evaluation performance on the validation set.

**Note:** Honestly, I forget which evaluation script to use for best performance. You can try both evaluation scripts to see which one works best for your setup. If they don't work, it shouldn't take too much time to debug.

## Useful Checkpoints

I've trained several checkpoints that you can use:

- `./finetune_parkinson/20250702_172406/` - Checkpoint from July 2nd, 2025
- `./finetune_parkinson/20250709_013137/` - Checkpoint from July 9th, 2025

**To check their performance:** Use TensorBoard to view the training logs. Navigate to the checkpoint folder and run:
```bash
cd ./finetune_parkinson/[checkpoint_folder]
tensorboard --logdir=tf_logs --port=6006 --host=0.0.0.0
```

This will show you the training curves, validation loss, and other metrics for each checkpoint.


