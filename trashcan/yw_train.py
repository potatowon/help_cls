import sys
import yaml
import torch
import numpy as np
import os
import monai
import argparse
# from yw_logger import setting
from yw_utils import set_seed, train_one_epoch, evaluate_one_epoch, batch_inference
from yw_dataset import load_data, load_test_data
from yw_transform import train_transforms, test_transforms
from types import SimpleNamespace
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
    
    def get_pretrained(self, model, pretrain_dir):
        # 1. pretrained 모델 불러오기
        pretrained_dict = torch.load(pretrain_dir, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. 존재하는 statedict만 overwrite
        model_dict.update(pretrained_dict)
        # 3. load statedict
        model.load_state_dict(model_dict)
        
        return model
    
    
def train_and_val(cfg, main_folder="ncp/workspace/blockstorage/kyw/dir_main"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # logger.info(f'model is {cfg.model}')
    # 모델 설정
    model = ClsModel().to(device)
    
    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0)
    
    # optimizer
    # if cfg.opt == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    # elif cfg.opt == 'admaw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
     
     # data load
    img_dir = '/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/pwi_TTP_in_MNI_brain.nii.gz'
    _, train_loader, _, valid_loader = load_data(img_dir, train_transforms, test_transforms, 16, test_size=0.3)
    
    print("Start training")
    
    # metrics 설정
    metrics = {
        "acc": {"value": 0, "filename": "acc_best_model.pth", "confusion_matrix" : 0},
        "f1": {"value": 0, "filename": "f1_best_model.pth", "confusion_matrix" : 0},
        "auroc": {"value": 0, "filename": "auroc_best_model.pth", "confusion_matrix" : 0}
    }
    
    for epoch in range(20):
        train_one_epoch(model, criterion, epoch, optimizer, train_loader, device)
        eval_acc, eval_f1, c_matrix, eval_auroc = evaluate_one_epoch(model, criterion, epoch, valid_loader, device)
        current_metrics = {"acc": eval_acc, "f1": eval_f1, "auroc": eval_auroc}
        
        # 좋은 가중치, 편향 값 저장
        for key in metrics:   
            if metrics[key]["value"] < current_metrics[key]:
                metrics[key]["value"] = current_metrics[key]
                metrics[key]['confusion_matrix'] = c_matrix
                save_path = os.path.join(main_folder, 'model', metrics[key]["filename"])
                torch.save(model.state_dict(), save_path)


# 추론 코드 
def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClsModel().to(device)
    model.load_state_dict(torch.load(cfg.infererence_pretrain_dir))
    img_dir = '/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/pwi_TTP_in_MNI_brain.nii.gz'
    _, test_loader = load_test_data(img_dir, test_transforms, 16)
    all_predictions = batch_inference(model, test_loader, device, cfg.model_path)

    # 예측된 확률에서 가장 높은 값의 인덱스를 가져와서 그에 해당하는 클래스 이름을 얻습니다.
    # 예시로, 클래스 이름을 'Type1', 'Type2', ... 라고 가정합니다.
    # 클래스 이름이 다르다면 이 부분을 수정해주세요.
    # 확률 추가
    predicted_classes = [f"Type{pred.argmax() + 1}" for pred in all_predictions]
    predicted_probabilities = [float(pred.max()) for pred in all_predictions]

    # ChallengeID 리스트 생성
    challenge_ids = [f"HT_Subject_{i:03}" for i in range(121, 201)]

    # DataFrame 생성
    df = pd.DataFrame({
        'ChallengeID': challenge_ids,
        'Submit_HTType': predicted_classes,
        'Probability': predicted_probabilities
    })
    workdir = '/home/ncp/workspace/blockstorage/kyw'
    # CSV로 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M_model")
    os.makedirs(os.path.join(workdir, 'infer_csv'), exist_ok=True)
    df.to_csv(os.path.join(workdir, 'infer_csv', 'predictions_'+66+'_'+current_time+'.csv'), index=False)

yaml_pth = './'

if __name__ == '__main__':
    # args, yaml 파일 가져오기

    # with open(yaml_pth, 'r') as stream:
    #     cfg = yaml.safe_load(stream)
    #     cfg = SimpleNamespace(**cfg)
    # seed 설정
    set_seed(66)
    # 모델과 로그 저장 폴더 생성
    # 현재의 날짜와 시간을 "YYYYMMDD_HHMM_model" 형식으로 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M_model")
    # main폴더 생성
    workdir = '/home/ncp/workspace/blockstorage/kyw'
    main_folder = os.path.join(workdir, current_time)
    os.makedirs(main_folder, exist_ok=True)
    # 'model'과 'log' 폴더 생성
    os.makedirs(os.path.join(main_folder, 'model'), exist_ok=True)
    os.makedirs(os.path.join(main_folder, 'log'), exist_ok=True)

    print(f"'{main_folder}' 아래에 'model'과 'log' 폴더에 데이터가 저장됩니다.")
    
    # with open(os.path.join(main_folder, 'log', 'cfg.yaml'), 'w') as f:
    #     yaml.dump(cfg, f)
    # logger 생성 장소 지정
    # logger = setting_logger(os.path.join(main_folder, 'log', 'logfile.txt'))