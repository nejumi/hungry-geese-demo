#!/usr/bin/env python
# coding: utf-8

import importlib.util
import sys
import os
from glob import glob
import numpy as np

def load_agent_from_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = agent_module
    spec.loader.exec_module(agent_module)
    return agent_module.agent

def state_to_board(state, config):
    rows, cols = config["rows"], config["columns"]
    board = np.full((rows, cols), -1, dtype=int)

    for index, goose in enumerate(state[0]["observation"]["geese"]):
        for pos in goose:
            row, col = pos // cols, pos % cols
            board[row][col] = index

    for pos in state[0]["observation"]["food"]:
        row, col = pos // cols, pos % cols
        board[row][col] = -2

    return board

def obs_to_agent_input(state, full_state):
    geese = full_state["observation"]["geese"]
    food = full_state["observation"]["food"]
    index = state["observation"]["index"]
    remaining_time = state["observation"]["remainingOverageTime"]

    return {
        "geese": geese,
        "food": food,
        "index": index,
        "remainingOverageTime": remaining_time
    }

def create_folders_if_not_exist(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_path_list(data_folder):
    folder_path = os.path.join(data_folder, '*.json')
    path_list = [path for path in glob(folder_path) if 'info' not in path]
    path_list = np.random.choice(path_list, len(path_list), replace=False)
    return path_list
