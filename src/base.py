
# This is a lightweight ML agent trained by self-play.
# After sharing this notebook,
# we will add Hungry Geese environment in our HandyRL library.
# https://github.com/DeNA/HandyRL
# We hope you enjoy reinforcement learning!
# Load PyTorch Model

import pickle
import bz2
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network for Hungry Geese

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self, layers, filters):
        super().__init__()
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}


# Input for Neural Network

def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

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

    return b.reshape(-1, 7, 11)


state_dict = pickle.loads(base64.b64decode(PARAM))
model = GeeseNet(layers, filters)
model.load_state_dict(state_dict)
model.eval()


# Main Function of Agent

obses = []

def agent(obs, _):
    obses.append(obs)
    x = make_input(obses)
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        o = model(xt)
    p = o['policy'].squeeze(0).detach().numpy()

    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    return actions[np.argmax(p)]
