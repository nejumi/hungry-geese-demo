#!/usr/bin/env python
# coding: utf-8
import base64
import pickle

import torch
from torch import nn
import torch.nn.functional as F

# Define the custom convolutional layer with torus topology
class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        # Calculate the edge size for padding
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        # Define the convolutional layer
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        # Optionally add batch normalization
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        # Pad the input tensor with the last rows and columns
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        # Apply the convolutional layer and batch normalization (if any)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h

# Define the neural network architecture
class GeeseNet(nn.Module):
    def __init__(self, layers, filters):
        super().__init__()
        # Define the input convolutional layer with torus topology
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        # Define the convolutional blocks with torus topology
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        # Define the policy head (linear layer with 4 outputs for the action probabilities)
        self.head_p = nn.Linear(filters, 4, bias=False)
        # Define the value head (linear layer with 1 output for the state value)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        # Apply the input convolutional layer with torus topology and ReLU activation
        h = F.relu_(self.conv0(x))
        # Apply the convolutional blocks with torus topology and ReLU activation
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # Compute the head positions
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        # Compute the average activations
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        # Compute the action probabilities and state value
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        return p, v

def model_weights_to_base64(model_path, layers, filters):
    model = GeeseNet(filters=filters, layers=layers)
    model = torch.load(model_path)
    model_data = pickle.dumps(model)
    encoded_weights = base64.b64encode(model_data).decode("utf-8")
    return encoded_weights

def create_submission_file(model_path, base_file_path, output_path, layers, filters):
    encoded_weights = model_weights_to_base64(model_path, layers, filters)
    with open(base_file_path, "r") as base_file:
        base_content = base_file.read()
    with open(output_path, "w") as f:
        f.write(f'layers, filters = {layers}, {filters}\n')
        f.write(f"PARAM = '{encoded_weights}'\n")
        f.write("\n")
        f.write(base_content)
