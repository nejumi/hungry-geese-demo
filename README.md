# hungry-geese-demo
## Project Title
- Imitation Learning and Hyperparameter Search with Wandb

## Overview
- This repository contains an imitation learning implementation to train an agent for the game ["Hungry Geese"](https://www.kaggle.com/c/hungry-geese) on Kaggle. This game requires a policy to navigate a game board with the aim of consuming food and avoiding collisions with the opponent's geese. We use a custom CNN model, named GeeseNet originally designed by yuricat and kyazuki, to capture spatial information from the game board.

## Prerequisites
- Wandb account (for tracking experiments and hyperparameter search)

## Repository Contents
- `training.py` - The main script to train the model. It includes functions for training and validation of the model, hyper-parameter parsing, and creating a DataLoader for training and validation sets.

- `get_data.py` - Script to download and extract the training dataset.

- `data_processing.py` - Contains functions to process the downloaded dataset and return train and validation sets.

- `model.py` - Defines the GeeseNet model and a function to create a submission file based on the trained model.

- `utils.py` - Contains helper functions like creating folders, getting path lists, etc.

- `visualization.py` - Contains a function to create a GIF from a submission for visualization purposes.

- `optuna_config_hungry_geese.yaml` - Sample configuration file for optimization by Launch with Optuna.

## Getting Started
You can start by cloning the repository:

```bash
git clone <repository-url>
cd <repository-dir>
```

Next, install the required Python packages. It's recommended to create a new Python environment, and once you activate the environment, you can install the packages using:

```bash
pip install -r requirements.txt
```

## Training
You can train the model by running the training.py script:

```bash
python training.py
```

You can provide the following optional arguments:

- `--layers`: number of layers in the `GeeseNet` model.
- `--filters`: number of filters in each convolutional layer in the `GeeseNet` model.
- `--batch_size`: the batch size used for training.
- `--data_folder`: the path of the folder where the training data is located.
- `--val_size`: the size of the validation set.
- `--n_epochs`: the number of epochs to train each chunk.
- `--chunk_size`: the number of samples in each chunk of the training data.
- `--chunk_num`: the number of chunks to be used for training.
- `--project`: project name for wandb.
- `--entity`: entity name for wandb.

For example:

```bash
python training.py --layers 16 --filters 16 --batch_size 4096
```

## Hyperparameter Optimization by Launch with Optuna
WandB Sweeps can used to optimize hyperparameters. This can also be done more scalably on Launch. Please refer to the [documentation](https://docs.wandb.ai/guides/launch/sweeps-on-launch) for details on the settings.
For example:

```bash
wandb launch-sweep optuna_config_hungry_geese.yaml -q "your-queue-name" -p your-project-name -e your-entity-name
```

## Results
The training script uses the [Weights & Biases](https://wandb.ai/site) (wandb) platform to track the model's performance. After each epoch, the script logs the loss, accuracy, and win rate of the model. It also logs a GIF of a self-match episode for the agent. You can visualize these results on the wandb platform.
![wandb_hungry_geese](https://github.com/nejumi/hungry-geese-demo/assets/24971026/a12deca1-9601-429d-b956-44917a138510)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Kaggle for providing the [Hungry Geese](https://www.kaggle.com/c/hungry-geese) environment.
- yuricat and cazuki for the GeeseNet and related implementations.
