#!/usr/bin/env python
# coding: utf-8
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

# Define the function to convert the observation to a neural network input tensor
def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]
    step = obs['step']

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs['index']) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs['index']) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs['index']) % 4, pos] = 1

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - obs['index']) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    b = b.reshape(-1, 7, 11)

    return b.astype('float32')

# Function to create a dataset from a JSON file
def create_best_dataset_from_json(filepath, json_object=None, standing=0):
    # Load the JSON data from a file or an object
    if json_object is None:
        json_open = open(filepath, 'r')
        json_load = json.load(json_open)
    else:
        json_load = json_object
    
    try:
        # Find the index of the winning geese
        winner_index = np.argmax(np.argsort(json_load['rewards']) == 3-standing)

        obses = []
        X = []
        y = []
        actions = {'NORTH':0, 'SOUTH':1, 'WEST':2, 'EAST':3}

        # Loop over the steps of the game
        for i in range(len(json_load['steps'])-1):
            # If the winning geese is still active
            if json_load['steps'][i][winner_index]['status'] == 'ACTIVE':
                # Get the action taken by the winning geese
                y_ = json_load['steps'][i+1][winner_index]['action']
                if y_ is not None:
                    # Get the observation of the winning geese
                    step = json_load['steps'][i]
                    step[winner_index]['observation']['geese'] = step[0]['observation']['geese']
                    step[winner_index]['observation']['food'] = step[0]['observation']['food']
                    step[winner_index]['observation']['step'] = step[0]['observation']['step']
                    obses.append(step[winner_index]['observation'])
                    y.append(actions[y_])        

        # Loop over the observations and create the input tensor
        for j in range(len(obses)):
            X_ = make_input(obses[:j+1])
            X.append(X_)

        # Convert the input tensor and labels to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.uint8)

        return X, y
    except:
        return 0, 0


# Function to create a dataset from a list of JSON file paths
def get_dataset(paths, n, chunk_size):
    # Calculate the number of chunks and select the chunk for this worker
    num_chunk = np.floor(len(paths)/chunk_size).astype(int)
    paths_chunk = paths[n*chunk_size:(n+1)*chunk_size]
    
    X_all = []
    p_all = []
    v_all = []
    rewards = [1, 0.33, -0.33, -1]
    
    # Loop over the standings
    for i in range(4):
        X_train = []
        y_train = []

        # Loop over the JSON file paths in the current chunk
        for path in tqdm(paths_chunk):
            # Create the input tensor and label from the JSON file
            X, y = create_best_dataset_from_json(path, standing=i)
            # If the input tensor is not empty, add it to the list
            if X is not 0:
                X_train.append(X)
                y_train.append(y)

        # Concatenate the input tensors and labels, remove duplicates and create the value labels
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_train, unique_index = np.unique(X_train, axis=0, return_index=True)
        y_train = y_train[unique_index]
        v_train = np.zeros(len(y_train))
        v_train[:] = rewards[i]

        # Add the input tensor, label and value to the lists
        X_all.append(X_train)
        p_all.append(y_train)
        v_all.append(v_train)

    # Concatenate the input tensors, labels and values and return the dataset
    X_all = np.concatenate(X_all)
    p_all = np.concatenate(p_all)
    v_all = np.concatenate(v_all)
    return X_all, p_all, v_all

def one_hot_numpy(input_array, num_classes):
    eye = np.eye(num_classes)
    return eye[input_array]

class GeeseDataset(Dataset):
    def __init__(self, X, p, v):
        self.X = X
        self.p = p
        self.v = v

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_batch = self.X[idx]
        p_batch = self.p[idx]
        v_batch = self.v[idx]

        # Apply data augmentation (random horizontal and vertical flips)
        if np.random.uniform(0,1) > 0.5:
            X_batch = np.flip(X_batch, axis=2).copy()
            p_batch = np.argmax(one_hot_numpy(p_batch, num_classes=4)[[0,1,3,2]])

        if np.random.uniform(0,1) > 0.5:
            X_batch = np.flip(X_batch, axis=1).copy()
            p_batch = np.argmax(one_hot_numpy(p_batch, num_classes=4)[[1,0,2,3]])

        return X_batch, p_batch, v_batch
    