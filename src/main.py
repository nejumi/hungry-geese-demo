#!/usr/bin/env python
# coding: utf-8

import wandb
from training import run_sweep
import yaml
import warnings
warnings.simplefilter('ignore')

def main():
    with open("config.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="geesenet-sweep-all-data-demo")
    wandb.agent(sweep_id, function=run_sweep)

if __name__ == "__main__":
    main()