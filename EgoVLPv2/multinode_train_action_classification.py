import os
import torch
import argparse
from trainer.trainer_action_classification import Multi_Trainer_Action_Classification
from parse_config import ConfigParser

def main():
    parser = argparse.ArgumentParser(description='Action Classification Training')
    parser.add_argument('--config', default='configs/ft/parkinson_action_classification.json', type=str, help='Config file path')
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    config = ConfigParser.from_args(args)
    trainer = Multi_Trainer_Action_Classification(
        model=config.initialize('model'),
        loss=config.initialize('loss'),
        metrics=[config.initialize('metrics')],
        optimizer=config.initialize('optimizer', config.initialize('model').parameters()),
        config=config,
        data_loader=config.initialize('data_loader'),
        valid_data_loader=config.initialize('valid_data_loader'),
        lr_scheduler=config.initialize('lr_scheduler', config.initialize('optimizer', config.initialize('model').parameters())),
        logger=None
    )
    trainer.train(args.local_rank)

if __name__ == '__main__':
    main() 