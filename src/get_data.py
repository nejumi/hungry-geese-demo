#!/usr/bin/env python
# coding: utf-8

import opendatasets as od
from utils import create_folders_if_not_exist

create_folders_if_not_exist(['../input'])
od.download('https://www.kaggle.com/datasets/nejumi/hungry-geese-episodes-created-by-self-play', data_dir='../input/')
