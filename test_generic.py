import os
import numpy as np
import pandas as pd
import sklearn
import torch
from ml_set import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, recall_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import precision_score, f1_score
from OMetaNet import *
import pickle
import config
import argparse

if __name__ == "__main__":

    # 设置设备：优先GPU，否则CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    # 加载配置并初始化模型
    cf = config.get_train_config()
    cf.task = 'test.csv'
    model = OMetaNet(cf)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='specify the path for the model to evaluate')
    args = parser.parse_args()

    # 读取测试数据
    _, test = data_read_all_U()
    print("len(test)", len(test))

    test_seq = test.iloc[:, 0]
    test_label = torch.tensor(np.array(test.iloc[:, 1], dtype='int64')).to(device, non_blocking=True)
    test_encoding = torch.tensor(np.load('./dataset/U/one_hot_ncp_features_test.npy')).to(device, non_blocking=True)
    test_embedding = torch.tensor(np.load('./dataset/U/KN_features_test.npy')).to(device, non_blocking=True)
    test_str_embedding = torch.tensor(
        np.load('./dataset/U/bert_embeddings_test.npy')).to(device, non_blocking=True)

    pt_files = "./model/OMetaNet-generic.pt"

    Result = []
    Result_softmax = []
    batch_size = cf.batch_size

    print('loading model ', pt_files)
    state_dict = torch.load(pt_files, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # 分批推理
    results = []
    with torch.no_grad():
        for i in range(0, len(test_encoding), batch_size):
            batch_encoding = test_encoding[i:i + batch_size]
            batch_embedding = test_embedding[i:i + batch_size]
            batch_str_embedding = test_str_embedding[i:i + batch_size]
            batch_result, _ = model(batch_encoding, batch_embedding, batch_str_embedding)
            results.append(batch_result)
        result = torch.cat(results, dim=0)
        result_softmax = F.softmax(result, dim=1)

    Result.append(result)
    Result_softmax.append(result_softmax)

    _, predicted = torch.max(result_softmax, 1)
    correct = (predicted == test_label).sum().item()
    result_np = result.cpu().detach().numpy()
    result_softmax_np = result_softmax.cpu().detach().numpy()

    tn, fp, fn, tp = confusion_matrix(test_label.cpu(), predicted.cpu()).ravel()
    test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    pre = precision_score(test_label.cpu(), predicted.cpu())
    f1 = f1_score(test_label.cpu(), predicted.cpu())
    mcc = matthews_corrcoef(test_label.cpu(), predicted.cpu())
    test_auc = roc_auc_score(test_label.cpu(), result_softmax_np[:, 1])
    ap = average_precision_score(test_label.cpu(), result_softmax_np[:, 1])

    result_str = (
        f"Model file name: {pt_files}  "
        f"Accuracy: {test_acc:.2f}%  "
        f"SEN: {sen * 100:.2f}%  SPEC: {spec * 100:.2f}%  "
        f"PRE: {pre * 100:.2f}%  F1: {f1 * 100:.2f}%  "
        f"MCC: {mcc * 100:.2f}%  AUC: {test_auc * 100:.2f}%  AP: {ap * 100:.2f}%\n"
    )
    print(result_str)

