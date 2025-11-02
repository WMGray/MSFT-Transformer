from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import os
from os.path import join as join
import torch
import numpy as np
import random
from sklearn.base import BaseEstimator
import pandas as pd
from typing import List, Any, Dict


def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.use_deterministic_algorithms(True)

# 评分指标
def my_auc(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    y_proba = net.predict_proba(X)
    return roc_auc_score(y, y_proba[:, 1])


def my_f1(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    y_proba = net.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=1)
    return f1_score(y, y_pred)

def check_sample_order(files: List[str]) -> None:
    """
    :param files: List of feature path
    :return:
    """
    samples = []
    for file in files:
        df = pd.read_csv(file)
        print(list(df['sample_id'])[:5])
        samples.append(list(df['sample_id']))

    for sample in samples:
        if samples[0] != sample:
            assert 0, "The order of samples is inconsistent across files."

def check_record(paras: Dict, df_path: str) -> bool:
    """
    :param paras: need to check
    :param res_df:
    :return:
    """
    if not os.path.exists(df_path):
        return True
    print(paras)
    res_df = pd.read_csv(df_path)[list(paras.keys())]

    for d in res_df.to_dict(orient='records'):
        if d == dict(paras):
            return False
    return True



def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> (dict[str, float], pd.DataFrame):
    y_true, y_pred = y, net.predict(X)
    y_prob = net.predict_proba(X)
    print(y_true.shape)
    print(y_prob.shape)
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob_0': y_prob[:, 0].squeeze(),  # 第一个类别的概率
        'y_prob_1': y_prob[:, 1].squeeze()  # 第二个类别的概率
    })
    try:
        y_true, y_pred = y, net.predict(X)
        y_prob = net.predict_proba(X)
        # 记录 预测值 和 准确值
        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        metrics = {
            'AUC': round(roc_auc_score(y_true, y_prob[:, 1]), 4),
            'ACC': round(accuracy_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'F1': round(f1_score(y_true, y_pred), 4)
        }
        return metrics, df

    except:
        return {
            'AUC': -1.0,
            'ACC': -1.0,
            'Recall': -1.0,
            'Precision': -1.0,
            'F1': -1.0
        }, pd.DataFrame({})
