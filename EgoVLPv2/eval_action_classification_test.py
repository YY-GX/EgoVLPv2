import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import model.loss as module_loss
import model.metric as module_metric
import model.model_epic_charades as module_arch
import data_loader.data_loader as module_data
from parse_config import ConfigParser

def run_test_evaluation(config_path, checkpoint_path, batch_size=16, device='cuda', save_csv=None, save_dir='.', resume=None):
    use_cuda = torch.cuda.is_available() and device.startswith('cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Load config directly from file
    from utils.util import read_json
    from pathlib import Path
    config_dict = read_json(Path(config_path))
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, config_dict):
            self._config = config_dict
            
        def __getitem__(self, key):
            return self._config[key]
            
        def initialize(self, name, module, *args, **kwargs):
            module_name = self._config[name]['type']
            module_args = dict(self._config[name]['args'])
            module_args.update(kwargs)
            return getattr(module, module_name)(*args, **module_args)
    
    config = SimpleConfig(config_dict)
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = batch_size
    config._config['data_loader']['args']['num_workers'] = 1
    # Force non-distributed dataloader for test set evaluation
    if config._config['data_loader']['type'].startswith('MultiDist') or config._config['data_loader']['type'].startswith('Dist'):
        config._config['data_loader']['type'] = 'TextVideoDataLoader'

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
    # Setup data_loader for test set
    test_data_loader = config.initialize('data_loader', module_data)
    # Build model
    model = config.initialize('arch', module_arch)
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if use_cuda else 'cpu')
    except RuntimeError as e:
        print(f"Warning: {e}\nRetrying checkpoint load with map_location='cpu'...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    from data_loader.ParkinsonEgo_dataset import ParkinsonEgo
    idx_to_label = ParkinsonEgo.IDX_TO_ACTION_LABEL
    label_to_idx = ParkinsonEgo.ACTION_LABEL_TO_IDX
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    with torch.no_grad():
        class_tokenized = tokenizer(class_names, return_tensors="pt", padding=True, truncation=True).to(device)
        class_text_embeds = model.compute_text(class_tokenized).float()
        class_text_embeds = F.normalize(class_text_embeds, dim=-1)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Evaluating'):
            video = batch['video'].to(device)
            text = {k: v.to(device) for k, v in batch['text'].items()}
            label = batch['label'].to(device)
            infer_out = model.infer({'video': video, 'text': text}, task_names='Dual')
            video_embeds = infer_out['video_embeds']
            # Use Euclidean distance for prediction, matching validation
            distances = torch.cdist(video_embeds, class_text_embeds)
            pred = torch.argmin(distances, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_paths.extend(batch['meta']['paths'])

    acc = correct / total if total > 0 else 0.0
    print(f'\nTest set accuracy: {acc*100:.2f}% ({correct}/{total})')

    updown_indices = [label_to_idx['upstair'], label_to_idx['downstair']]
    updown_total = 0
    updown_correct = 0
    for true_idx, pred_idx in zip(all_labels, all_preds):
        if true_idx in updown_indices:
            updown_total += 1
            if true_idx == pred_idx:
                updown_correct += 1
    updown_acc = updown_correct / updown_total if updown_total > 0 else 0.0
    print(f'Upstair+Downstair accuracy: {updown_acc*100:.2f}% ({updown_correct}/{updown_total})')

    if save_csv:
        import csv
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'true_label', 'pred_label'])
            for path, true_idx, pred_idx in zip(all_paths, all_labels, all_preds):
                writer.writerow([path, idx_to_label[true_idx], idx_to_label[pred_idx]])
        print(f'Predictions saved to {save_csv}')

    return {
        'test_accuracy': acc,
        'test_correct': correct,
        'test_total': total,
        'updown_accuracy': updown_acc,
        'updown_correct': updown_correct,
        'updown_total': updown_total
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate action classification on test set')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--save_csv', default=None, type=str, help='Path to save predictions as CSV (optional)')
    parser.add_argument('--save_dir', default='.', type=str, help='Directory for model saving (required by config system)')
    parser.add_argument('--resume', default=None, type=str, help='Path to resume checkpoint (not used for test, but required by config system)')
    args = parser.parse_args()
    run_test_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        save_csv=args.save_csv,
        save_dir=args.save_dir,
        resume=args.resume
    ) 