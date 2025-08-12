import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import model.model_epic_charades as module_arch
import data_loader.data_loader as module_data
from parse_config import ConfigParser

def run_valstyle_test(config_path, checkpoint_path, batch_size=16, device='cuda', save_csv=None, save_dir='.', resume=None):
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

    # Pre-compute action text embeddings
    with torch.no_grad():
        action_tokens = tokenizer(class_names, padding=True, truncation=True, return_tensors='pt').to(device)
        action_embeddings = model.compute_text(action_tokens).float()

    all_predictions = []
    all_labels = []
    all_paths = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='ValStyle Test Evaluating'):
            video = batch['video'].to(device)
            text = {k: v.to(device) for k, v in batch['text'].items()}
            label = batch['label'].to(device)
            # Use the same infer logic as validation
            ret = model.infer({'video': video, 'text': text}, return_embeds=True, task_names="Dual", ret={})
            video_embeds = ret['video_embeds']
            distances = torch.cdist(video_embeds, action_embeddings)
            predictions = torch.argmin(distances, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(label.cpu())
            all_paths.extend(batch['meta']['paths'])
            correct += (predictions == label).sum().item()
            total += label.size(0)

    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = correct / total if total > 0 else 0.0
        print(f'ValStyle Test set accuracy: {acc*100:.2f}% ({correct}/{total})')
        if save_csv:
            import csv
            with open(save_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['video_path', 'true_label', 'pred_label'])
                for path, true_idx, pred_idx in zip(all_paths, all_labels, all_predictions):
                    writer.writerow([path, idx_to_label[true_idx], idx_to_label[pred_idx]])
            print(f'Predictions saved to {save_csv}')
        return {
            'test_accuracy': acc,
            'test_correct': correct,
            'test_total': total
        }
    else:
        print('No predictions made.')
        return {
            'test_accuracy': 0.0,
            'test_correct': 0,
            'test_total': 0
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ValStyle Test: Evaluate action classification on test set (validation logic)')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--save_csv', default=None, type=str, help='Path to save predictions as CSV (optional)')
    parser.add_argument('--save_dir', default='.', type=str, help='Directory for model saving (required by config system)')
    parser.add_argument('--resume', default=None, type=str, help='Path to resume checkpoint (not used for test, but required by config system)')
    args = parser.parse_args()
    run_valstyle_test(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        save_csv=args.save_csv,
        save_dir=args.save_dir,
        resume=args.resume
    ) 