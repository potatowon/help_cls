# 1. 데이터 읽어오기
# 2. 데이터 트랜스폼 하기
# 22. 모델 불러오기 --> 이미 가중치 학습된거 불러오기 --> 학습하면서 더 좋은 모델 저장하기
# 3. 트레인 하기
    # 3-1 loss설정
    # 3-2 optim 설정
    # 3-3 metrics 설정
    # 3-0 배치사이즈, lr,
# 4. 저장된 모델 불러와서 테스트 하기
# 5. 테스트 한 값 저장하기

import pandas as pd
import glob
import torch
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, ImageDataset



# 1. data load

def load_data(img_path, train_transforms, valid_transforms, batch_size, test_size=0.3):
    df = pd.read_csv('/workspace/blockstorage/kyw/med_classification/StudyHT_Open.csv')
    labels = list(map(int, list(df.loc[:119, 'GT_HTType']))) # label 
    
    # load image data
    mni_img_dirs = glob.glob(img_path)
    img_dirs = sorted(mni_img_dirs) # train ~119 
    train_img_dirs = img_dirs[:120]
    print(len(train_img_dirs))
    print(len(labels))
    # split train and valid by filenames 파일명으로 나눈거임
    train_img_dirs, valid_img_dirs, train_labels, valid_labels = train_test_split(train_img_dirs,
                                                                                  labels,
                                                                                  test_size=test_size,
                                                                                  stratify=labels)
    
    # imagedataload, dataload 이용해서 사용할 loader생성하기
    train_dataset = ImageDataset(image_files=train_img_dirs, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    
    valid_dataset = ImageDataset(image_files=valid_img_dirs, labels=valid_labels, transform=valid_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    
    print('finished data load')
    return train_dataset, train_loader, valid_dataset, valid_loader
    
# 2. transform define

import monai
import numpy as np
from monai.transforms import (
    Compose,
    AddChannel,
    Spacing,
    ScaleIntensity,
    NormalizeIntensityd,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandGaussianNoised,   
    ToTensord,
    Resize,
    EnsureChannelFirst
)

train_transforms = Compose([ScaleIntensity(), 
                            EnsureChannelFirst(), 
                            Resize((96, 96, 96)),
                            ])


val_transforms = Compose([ScaleIntensity(), 
                          EnsureChannelFirst(), 
                          Resize((96, 96, 96)),
                          ])
img_path = "/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/dwi_adc_in_MNI_brain.nii.gz"
train_dataset, train_loader, valid_dataset, valid_loader = load_data(img_path, train_transforms, val_transforms, 8)

# 3. model 불러오기
'''
미리 학습된 모델을 사용할 예정임. --> 일단 모델의 아키텍쳐를 불러오고
미리 학습된 가중치를 매칭해줌 --> 미리 학습된 아키텍쳐 확인하니까 2진분류짜리임 따로 분류단계 만들어줄 필요 없음

'''

import monai
from resnet3d import ResNet3dClassification
from resnet3d_pretrained import *
import torch

def get_pretrained(model, pretrain_dir):
    # 학습된 모델의 객체를 로드합니다.
    pretrained_dict = torch.load(pretrain_dir, map_location=torch.device('cpu'))
    # 기존 모델의 가중치, 편향을 딕셔너리 형태로 로드합니다.
    model_dict = model.state_dict()
    # 학습된 모델의 레이어가 기존 모델에 있는 경우에만 사용해야 하므로 학습된 모델의 
    # 형태를 사용할 모델에 모양에 맞게 바꿔줍니다.
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 이제 이 가중치의모양은 원래 모델과 같으므로 덮어씌울 수 있습니다.
    model_dict.update(pretrained_dict)
    # 덮어진 가중치를 모델에 업데이트 합니다.
    model.load_state_dict(model_dict)
    
    
    print('weight bias update complete')    
    
    return model

model = resnet101(shortcut_type='B')
pretrain_dir = '/workspace/blockstorage/kyw/pretrainedmodel/resnet_101.pth'

premodel = get_pretrained(model, pretrain_dir)

# Do it Train!!!!!!!!!!!!!!!!!!
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryAUROC
)

# 1 epoch train 
def train_one_epoch(model, criterion, epoch, optimizer, data_loader, device):

    model.train()
    
    train_loss = 0
    total = 0
    
    acc_metric = BinaryAccuracy()
    f1_metric = BinaryF1Score()
    # auroc = BinaryAUROC()
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # forward 
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()

        # backward
        optimizer.step()
        train_loss += loss.item()
        
        total += targets.size(0) # batchsize
        # preds = preds.cpu()
        # targets = targets.cpu()
  
        preds = preds.softmax(dim=-1) # probablilty
        preds = preds.argmax(dim=-1) # prediction index 
        
        acc = acc_metric(preds, targets)
        f1 = f1_metric(preds, targets)
        # auroc = auroc(preds, targets)

    acc = acc_metric.compute() # 최종값의 계산
    f1 = f1_metric.compute()
    # auroc = auroc.compute()

    # logger.debug(f'Epoch {epoch:<3} ,train_Loss = {train_loss / total :<8}, train_acc = {acc:<8}, train_f1 = {f1:<8}, train_auroc = {auroc:<8}')
    print(f'Epoch {epoch:<3} ,train_Loss = {train_loss / total :<8}, train_acc = {acc:<8}, train_f1 = {f1:<8}')
    acc_metric.reset()
    f1_metric.reset()
    
def evaluate_one_epoch(model, criterion, epoch, valid_loader, device):
    print('\n[ Test epoch: %d ]' % epoch)
    model.eval()
    valid_loss = 0
    total = 0
    
    acc_metric = BinaryAccuracy()
    f1_metric = BinaryF1Score()
    # confmat = BinaryConfusionMatrix()
    auroc = BinaryAUROC(thresholds=None)
    
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            preds = model(inputs)
            valid_loss += criterion(preds, targets).item()
            
            preds = preds.softmax(dim=-1)
            preds = preds.argmax(dim=-1)
            acc = acc_metric(preds, targets)
            f1 = f1_metric(preds, targets)
            # c_matrix = confmat(preds, targets)
            # auroc= auroc(preds, targets)
            
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        #auroc = auroc.compute()
        # logger.debug(f'Epoch {epoch:<3} ,valid_Loss = {valid_loss / total :<8}, valid_acc = {acc:<8}, valid_f1 = {f1:<8}')
            
        acc_metric.reset()
        f1_metric.reset()
        # confmat.reset()
        #auroc.reset()
        
    return acc, f1
    
import torch.nn as nn
import os
# 괜찮은 척도를 가진 얘들은 저장을 해놓고 그걸로 추론할거임
# 추론할 모델 저장하는 경로를 설정하자. v--> mainfolder save
def train(model, train_loader, valid_loader, epochs):
    '''
    model save path :
    use pretrained model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    # metrics 설정
    metrics = {
        "acc": {"value": 0, "filename": "acc_best_model.pth", "confusion_matrix" : 0},
        "f1": {"value": 0, "filename": "f1_best_model.pth", "confusion_matrix" : 0},
    }
    print('start train')
    for epoch in range(epochs):
        train_one_epoch(model, criterion, 10,  optimizer, train_loader, device)
        eval_acc, eval_f1 = evaluate_one_epoch(model, criterion, epoch, valid_loader, device)
        current_metrics = {"acc": eval_acc, "f1": eval_f1}
        # 좋은 가중치, 편향 값 저장
    for key in metrics:   
        if metrics[key]["value"] < current_metrics[key]:
            metrics[key]["value"] = current_metrics[key]
            save_path = os.path.join('/workspace/blockstorage/kyw/mainfolder', 'model', metrics[key]["filename"])
            torch.save(model.state_dict(), save_path)
            
            
train(premodel, train_loader, valid_loader, 10)
    
    
