import os
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys
import argparse
import collections
import transformers
import signal
import subprocess
from set_optim_schedule import set_schedule
import yaml
import datetime
import shutil
import json
import tempfile

import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_epic_charades as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer.trainer_epic import Multi_Trainer_dist_MIR
from utils.util import replace_nested_dict_item
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
# Specifically suppress torchvision deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._functional_video")


parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('--task_names', default='EgoNCE_ITM_MLM', type=str, help='Task_Names')
parser.add_argument('-c', '--config', default='/fsx/spraman3/Video_Language_Pretraining/Pre-training/EgoVLP_multinode/configs/pt/egoclip.json', type=str,
                help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                help='indices of GPUs to enable (default: all)')
parser.add_argument('-o', '--observe', action='store_true',
                help='Whether to observe (neptune)')
parser.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
parser.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
parser.add_argument('-sc', '--schedule', default=[60, 80])
parser.add_argument('--print_freq', type=int, default=100, help="print loss after this number of steps")
parser.add_argument('--save_dir', type=str, help="dirctory for model saving")
parser.add_argument('--annotation_mode', type=str, default='mode2', choices=['mode1', 'mode2'], help='Annotation split mode to use (mode1 or mode2)')
parser.add_argument('--debug_eval', action='store_true', help='If set, run test+val eval and save predictions after best ckpt is saved')

CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
# config will be initialized after we create the temporary config file


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def main():
    
    args = parser.parse_args()
    # Create datetime subfolder under save_dir
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_dir = args.save_dir
    subfolder = os.path.join(base_save_dir, now)
    os.makedirs(subfolder, exist_ok=True)
    args.save_dir = subfolder
    # Save config/hyperparams to subfolder
    config_copy_path = os.path.join(subfolder, 'config.json')
    shutil.copy(args.config, config_copy_path)
    # Pass annotation_mode into config and format meta_dir
    with open(args.config, 'r') as f:
        config_json = json.load(f)
    config_json['annotation_mode'] = args.annotation_mode
    meta_dir = config_json['data_loader']['args']['meta_dir']
    print(f"Original meta_dir: {meta_dir}")
    if '{annotation_mode}' in meta_dir:
        config_json['data_loader']['args']['meta_dir'] = meta_dir.replace('{annotation_mode}', args.annotation_mode)
        print(f"Updated meta_dir: {config_json['data_loader']['args']['meta_dir']}")
    else:
        print(f"No placeholder found in meta_dir: {meta_dir}")
    # Save the updated config to a temp file for this run
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmpf:
        json.dump(config_json, tmpf, indent=4)
        tmp_config_path = tmpf.name
    args.config = tmp_config_path
    
    # Now initialize the config parser with the updated config file
    global config
    config = ConfigParser(parser)
    
    # Define tensorboard log dir in subfolder
    tf_log_dir = os.path.join(subfolder, 'tf_logs')
    args.ngpus_per_node = torch.cuda.device_count()
    # Get port from environment variable or use default
    port = 58600  # 58400 default
    # port = 58601  # 58400 default
    
    if 'SLURM_JOB_ID' in os.environ:
        print("If is being executed")
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:{port}'
    else:
        print("Else is being executed")
        args.rank = 0
        args.dist_url = f'tcp://localhost:{port}'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args, tf_log_dir, config), args.ngpus_per_node)


def main_worker(gpu, args, tf_log_dir, config):
    import yaml
    with open('./EgoNCE_MLM_ITM_Config.yml') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    print("main worker started")
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    print("init processed finished")

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    print("device set")

    
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None
        
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    
    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        print('Val dataset: ', [len(x.dataset) for x in valid_data_loader], ' samples')
    # build model architecture, then print to console
    
    print("dataloader initialized")

    
    model = config.initialize('arch', module_arch)

    if args.rank == 0:
        logger.info(model)

    
    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    # Note: metrics are not used in this trainer - validation uses custom distance-based evaluation
    metrics = []  # Empty list since we don't use standard metrics
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    
    ## set_schedule will be modified; currently same as pretraining
    max_steps = int(len(data_loader[0]) * config['trainer']['epochs'])
    if max_steps==0:
        max_steps = int(len(data_loader[0]) * 10)
    warmup_steps = config_yaml["warmup_steps"]
    if isinstance(config_yaml["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    optimizer, scheduler = set_schedule(model, config, config_yaml, max_steps, warmup_steps)
    
    lr_scheduler = None
    writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=tf_log_dir)

    print("trainer should being here")
    trainer = Multi_Trainer_dist_MIR(args, model, loss, metrics, optimizer, scheduler, gpu,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])


    print("trainer should being here")
    trainer.train(gpu)
    
    # --- Test evaluation after training ---
    import copy
    test_config = copy.deepcopy(config)
    if "type" in test_config["data_loader"] and "args" in test_config["data_loader"]:
        test_config['data_loader']['args'] = replace_nested_dict_item(test_config['data_loader']['args'], 'split', 'test')
        # test_config['data_loader']['args'] = replace_nested_dict_item(test_config['data_loader']['args'], 'num_workers', 1)
        # test_config['data_loader']['args'] = replace_nested_dict_item(test_config['data_loader']['args'], 'batch_size', 4)
        
        # Create test dataloader WITHOUT distributed sampling (same as validation)
        from data_loader.data_loader import dataset_loader
        from data_loader.transforms import init_video_transform_dict
        from torch.utils.data import DataLoader
        
        dataset_name = test_config['data_loader']['args']['dataset_name']
        text_params = test_config['data_loader']['args']['text_params']
        video_params = test_config['data_loader']['args']['video_params']
        data_dir = test_config['data_loader']['args']['data_dir']
        meta_dir = test_config['data_loader']['args']['meta_dir']
        split = test_config['data_loader']['args']['split']
        reader = test_config['data_loader']['args']['reader']
        batch_size = test_config['data_loader']['args']['batch_size']
        num_workers = test_config['data_loader']['args']['num_workers']
        shuffle = test_config['data_loader']['args']['shuffle']
        
        tsfm_dict = init_video_transform_dict()
        tsfm = tsfm_dict[split]
        test_dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, reader=reader)
        
        test_data_loader = [DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )]
        
    elif isinstance(test_config["data_loader"], list):
        new_cfg_li = []
        for dl_cfg in test_config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'test')
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'num_workers', 1)
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'batch_size', 4)
            new_cfg_li.append(dl_cfg)
        test_config._config['data_loader'] = new_cfg_li
        test_data_loader = [test_config.initialize('data_loader', module_data, index=idx) for idx in range(len(test_config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    # Temporarily set the trainer's valid_data_loader to test_data_loader and run _valid_epoch
    original_valid_data_loader = trainer.valid_data_loader
    trainer.valid_data_loader = test_data_loader
    test_log = trainer._valid_epoch(epoch=0, gpu=gpu)
    trainer.valid_data_loader = original_valid_data_loader
    if args.rank == 0:
        print("[TEST] Test set evaluation results:", test_log)

    if args.rank == 0:
        from torch.utils.data import ConcatDataset, DataLoader
        from data_loader.data_loader import dataset_loader
        from data_loader.transforms import init_video_transform_dict
        # Prepare val and test datasets
        val_config = copy.deepcopy(config)
        val_config['data_loader']['args'] = replace_nested_dict_item(val_config['data_loader']['args'], 'split', 'val')
        dataset_name = val_config['data_loader']['args']['dataset_name']
        text_params = val_config['data_loader']['args']['text_params']
        video_params = val_config['data_loader']['args']['video_params']
        data_dir = val_config['data_loader']['args']['data_dir']
        meta_dir = val_config['data_loader']['args']['meta_dir']
        split = val_config['data_loader']['args']['split']
        reader = val_config['data_loader']['args']['reader']
        batch_size = val_config['data_loader']['args']['batch_size']
        num_workers = val_config['data_loader']['args']['num_workers']
        shuffle = False
        tsfm_dict = init_video_transform_dict()
        tsfm_val = tsfm_dict['val']
        tsfm_test = tsfm_dict['test']
        val_dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, 'val', tsfm_val, reader=reader)
        test_dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, 'test', tsfm_test, reader=reader)
        combined_dataset = ConcatDataset([val_dataset, test_dataset])
        combined_loader = [DataLoader(
            dataset=combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )]
        # Use the new _valid_epoch with return_predictions
        trainer.valid_data_loader = combined_loader
        predictions = trainer._valid_epoch(epoch=0, gpu=gpu, return_predictions=True)
        with open(os.path.join(args.save_dir, "all_predictions.json"), "w") as f:
            import json
            json.dump(predictions, f, indent=2)
        print(f"Saved all predictions to {os.path.join(args.save_dir, 'all_predictions.json')}")


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        
        # Store original split
        original_split = config['data_loader']['args'].get('split', 'train')
        
        # Change split to 'val' for validation
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        
        # # Use smaller batch size and fewer workers for validation
        # config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'num_workers', 1)
        # config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'batch_size', 4)
        
        # Create validation dataloader WITHOUT distributed sampling
        # We'll create it manually to avoid the DistributedSampler
        from data_loader.data_loader import TextVideoDataLoader
        from torch.utils.data import DataLoader
        
        # Create the dataset directly
        dataset_name = config['data_loader']['args']['dataset_name']
        text_params = config['data_loader']['args']['text_params']
        video_params = config['data_loader']['args']['video_params']
        data_dir = config['data_loader']['args']['data_dir']
        meta_dir = config['data_loader']['args']['meta_dir']
        split = config['data_loader']['args']['split']
        reader = config['data_loader']['args']['reader']
        batch_size = config['data_loader']['args']['batch_size']
        num_workers = config['data_loader']['args']['num_workers']
        shuffle = config['data_loader']['args']['shuffle']
        
        # Create dataset
        from data_loader.data_loader import dataset_loader
        from data_loader.transforms import init_video_transform_dict
        
        tsfm_dict = init_video_transform_dict()
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, reader=reader)
        
        # Create non-distributed dataloader
        valid_data_loader = [DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )]
        
        # Restore original split
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', original_split)
        
    elif isinstance(config["data_loader"], list):
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            # Use smaller batch size and fewer workers for validation
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'num_workers', 1)
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'batch_size', 4)
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    main()
