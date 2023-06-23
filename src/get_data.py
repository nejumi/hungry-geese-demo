#!/usr/bin/env python
# coding: utf-8
import wandb
import zipfile
from utils import create_folders_if_not_exist

def download_and_extract_dataset(data_folder='../input'):
    create_folders_if_not_exist([data_folder])

    artifact_data = wandb.use_artifact('yuya-yamamoto/hungry-geese-dataset/hungry-geese-episodes:v1', type='dataset')
    zip_path = artifact_data.get_path('hungry-geese-episodes.zip').download()
    artifact_base = wandb.use_artifact('yuya-yamamoto/hungry-geese-dataset/hungry-geese-basefile:v0', type='code')
    base_path = artifact_base.get_path('base.py').download()

    # unzip dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    
    return zip_path, base_path
