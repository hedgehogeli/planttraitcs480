from bookkeeping import *
from data_processing import *

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import os

import timm


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, input_size=163, num_heads=4):
        super(MultiheadAttentionBlock, self).__init__()

        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_size)
        
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(input_size)
        

    def forward(self, x):
        attn_output1, _ = self.multihead_attn1(x, x, x)
        x = self.norm1(x + attn_output1)  
        
        attn_output2, _ = self.multihead_attn2(x, x, x)
        x = self.norm2(x + attn_output2)  

        return x


class BigBoyV3(nn.Module):
    def __init__(self):
        super(BigBoyV3, self).__init__()

        # self.image_processor = timm.create_model('mobilenetv2_140', pretrained=True)
        self.image_processor = timm.create_model('mobilenetv3_large_100', pretrained=True)
        
        self.data_processor = MultiheadAttentionBlock()

        concat_size = img_output_size + fc_output_size

        self.concat_attn = nn.MultiheadAttention(embed_dim=concat_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(concat_size)

        self.linear_head = nn.Sequential(
            nn.Linear(concat_size, concat_size), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(concat_size, num_predictions),
        )
    
    def forward(self, img, data):
        img_result = self.image_processor(img)
        data_result = self.data_processor(data)
        combined = torch.cat((img_result, data_result), dim=1)

        attn_output, _ = self.concat_attn(combined, combined, combined)
        combined = self.norm(combined + attn_output) 

        out = self.linear_head(combined)

        return out


img_output_size = 1000
fc_output_size = 256
num_predictions = 6


class BigBoyV2(nn.Module):
    def __init__(self):
        super(BigBoyV2, self).__init__()

        self.image_processor = timm.create_model('mobilenetv2_140', pretrained=True)
        # self.image_processor = timm.create_model('efficientnet_b1', pretrained=True)
        
        self.fc = FC()

        concat_size = img_output_size + fc_output_size
        self.linear_head = nn.Sequential(
            nn.Linear(concat_size, concat_size//2), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(concat_size//2, concat_size//4), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(concat_size//4, num_predictions),
            # nn.Sigmoid()
        )
    
    def forward(self, img, data):
        img_result = self.image_processor(img)
        data_result = self.fc(data)
        combined = torch.cat((img_result, data_result), dim=1)

        return self.linear_head(combined)
    


# learn_rate = 0.0005
# learn_decay = 0.9

    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(163, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),

            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),

            nn.Linear(256, fc_output_size), nn.ReLU(inplace=True), nn.Dropout(0.2),

            # nn.Linear(128, fc_output_size), nn.ReLU(inplace=True), nn.Dropout(0.2),

            # nn.Linear(256, fc_output_size)
        )

    def forward(self, input):
        return self.linear(input)


    