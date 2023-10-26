from yw_dataset import load_data, load_test_data
from yw_transform import test_transforms, train_transforms

import sys
import yaml
import torch
import numpy as np
import os
import monai

from torch import nn
from datetime import datetime
import pandas as pd
from collections import Counter
import monai
from resnet3d import ResNet3dClassification
from resnet3d_pretrained import *
import torch


class ClsModel:
    def __init__(self) -> None:
        torch.cuda.is_available()
        self.resnet101_model = self.get_pretrained(resnet101(shortcut_type='B'), '/home/ncp/workspace/blockstorage/kyw/pretrainedmodel/resnet_101.pth')
    
    def get_pretrained(self, model, pretrain_dir='/home/ncp/workspace/blockstorage/kyw/pretrainedmodel/resnet_101.pth'):
        # 1. pretrained 모델 불러오기
        pretrained_dict = torch.load(pretrain_dir, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. 존재하는 statedict만 overwrite
        model_dict.update(pretrained_dict)
        # 3. load statedict
        model.load_state_dict(model_dict)
        
        return model
# pretrain_dir = '/home/ncp/workspace/blockstorage/kyw/pretrainedmodel/resnet_101.pth'
img_dir  = '/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/pwi_TTP_in_MNI_brain.nii.gz'
def train_val(img_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClsModel().to(device)
    
    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0)
    
    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
     
     # data load
    # img_dir = '/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/pwi_TTP_in_MNI_brain.nii.gz'
    _, train_loader, _, valid_loader = load_data(img_dir, train_transforms, test_transforms, 16, test_size=0.3)
    
    print('start training')


if __name__ == '__main__':
    train_val(img_dir)