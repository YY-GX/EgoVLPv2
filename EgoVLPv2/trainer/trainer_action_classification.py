# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist

from base import BaseTrainer, Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
from pathlib import Path
import sys
import json

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )

class Multi_Trainer_Action_Classification(Multi_BaseTrainer_dist):
    """
    Trainer class for action classification
    """

    def __init__(self, args, model, loss, metrics, optimizer, scheduler, gpu, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, scheduler, gpu, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch, scaler, gpu):
        """
        Training logic for an epoch
        """
        if dist.get_rank() == 0:
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(self.args.save_dir) / str('stats.txt'), 'a', buffering=1)
            print(' '.join(sys.argv))
            print(' '.join(sys.argv), file=stats_file)

        log = {}

        if self.do_validation and epoch == 1:
            val_log = self._valid_epoch(0, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # Move data to GPU
                for k, v in data.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if torch.is_tensor(vv):
                                data[k][kk] = vv.cuda(gpu, non_blocking=True)
                    elif torch.is_tensor(v):
                        data[k] = v.cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.loss(output, data['label'])

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                if dist.get_rank()==0:
                    if (batch_idx) % self.args.print_freq == 0:
                        stats = dict(epoch=epoch, step=batch_idx,
                                lr_weights=self.optimizer.param_groups[0]['lr'],
                                loss=loss.item())
                        print(json.dumps(stats), file=stats_file)

                if self.writer is not None and self.args.rank == 0:
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.args.print_freq == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))

                self.optimizer.zero_grad()

            if batch_idx == self.len_epoch:
                break

        log.update({f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))})

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch, gpu):
        """
        Validate after training an epoch
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        preds = []
        labels = []
        
        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    try:
                        # Move data to GPU
                        for k, v in data.items():
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    if torch.is_tensor(vv):
                                        data[k][kk] = vv.cuda(gpu, non_blocking=True)
                            elif torch.is_tensor(v):
                                data[k] = v.cuda(gpu, non_blocking=True)

                        # Forward pass
                        output = self.model(data)
                        loss = self.loss(output, data['label'])

                        # Store predictions and labels
                        preds.append(output.detach().cpu())
                        labels.append(data['label'].detach().cpu())

                        total_val_loss[dl_idx] += loss.item()

                    except Exception as e:
                        print(f"Error processing validation batch {batch_idx} in dataloader {dl_idx}: {str(e)}")
                        print(f"Full traceback:", e.__traceback__)
                        raise

        # Concatenate all predictions and labels
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # Compute metrics
        metrics = self._eval_metrics((preds, labels))
        for i, metric in enumerate(self.metrics):
            total_val_metrics[0][i] = metrics[i]

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.valid_data_loader)):
                tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)

        return {
            'val_loss': total_val_loss[0] / len(self.valid_data_loader[0]),
            'val_metrics': total_val_metrics[0].tolist()
        }

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = len(self.data_loader[dl_idx])
        return base.format(current, total, 100.0 * current / total) 