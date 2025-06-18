import torch
from .trainer_epic import Multi_Trainer_dist_MIR
from model.metric import oscc_metrics

class Multi_Trainer_Action_Classification(Multi_Trainer_dist_MIR):
    def _valid_epoch(self, epoch, gpu):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                # Move tensors to GPU
                for k, v in data.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if torch.is_tensor(vv):
                                data[k][kk] = vv.cuda(gpu, non_blocking=True)
                    elif torch.is_tensor(v):
                        data[k] = v.cuda(gpu, non_blocking=True)
                # Forward pass
                output = self.model(data)
                # Assume output is logits, and data['label'] is ground-truth
                # (You may need to adjust this depending on your model's output)
                preds.append(output.detach().cpu())
                labels.append(data['label'].detach().cpu())
        # Concatenate all predictions and labels
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        # Compute accuracy
        metrics = oscc_metrics(preds, labels)
        print(f"Validation Accuracy: {metrics['accuracy']:.2f}%")
        return metrics 