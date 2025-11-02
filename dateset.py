from collections import Counter
import os
from os.path import join as join
from typing import Tuple, Dict, Any

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDict
from typing import List
#########################################
# 待修改
from utils import setup_seed, check_sample_order


class dataset(Dataset):
    def __init__(self, path: str, use_cols=None, normalize=True, sort=False):
        super().__init__()
        if use_cols is None:
            use_cols = []
        print(f'Load dataset from {path}')

        if use_cols:
            self.data = pd.read_csv(path)[use_cols]
        else:
            self.data = pd.read_csv(path)
        if sort:
            print("开香槟啦， 要重排序")
            self.data.sort_values('sample_id', inplace=True)  # sort

        self.label = self.data.iloc[:, 1].values.squeeze()
        self.data = self.data.iloc[:, 2:].values

        if normalize:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def load_single_features(seed: int, disease: str, feature: List):
    """
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    print(feature)
    path = f"./Data/{disease}/{feature}_abundance.csv"

    data = dataset(path, use_cols=None)  # Z-Score

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(data.data, data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=data.label)
    # 合并两个输入 -- Skorch
    x_train = SliceDict(f1_input=x_train_ko.astype(np.float32))
    x_test = SliceDict(f1_input=x_test_ko.astype(np.float32))

    y_train, y_test = y_train_ko, y_test_ko
    print(Counter(y_train), Counter(y_test))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

    return x_train, x_test, y_train, y_test


def load_full_features(seed: int, disease: str, feature: List, sort: bool=False, noise: float = 0):
    """
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    print(feature)
    if noise:
        f1_path = f"./Data/{disease}/{feature[0]}_noisy_{noise}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_noisy_{noise}_abundance.csv"
    else:
        f1_path = f"./Data/{disease}/{feature[0]}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])

    f1_data = dataset(f1_path, use_cols=None, sort=sort)  # Z-Score
    f2_data = dataset(f2_path, use_cols=None, sort=sort)

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(f1_data.data, f1_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f1_data.label)
    x_train_go, x_test_go, y_train_go, y_test_go = train_test_split(f2_data.data, f2_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f2_data.label)
    # 合并两个输入 -- Skorch
    if (y_train_ko.all() == y_train_go.all()) and (y_test_ko.all() == y_test_go.all()):
        x_train = SliceDict(f1_input=x_train_ko.astype(np.float32), f2_input=x_train_go.astype(np.float32))
        x_test = SliceDict(f1_input=x_test_ko.astype(np.float32), f2_input=x_test_go.astype(np.float32))

        y_train, y_test = y_train_ko, y_test_ko
        print(Counter(y_train), Counter(y_test))

        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

        return x_train, x_test, y_train, y_test
    else:
        assert 0, "两个特征的标签不匹配"


def load_features(disease: str, features: List):
    fps = [
        f"./Data/{disease}/{features[0]}_abundance.csv",
        f"./Data/{disease}/{features[1]}_abundance.csv"
    ]  # 根据特征名字提取路径

    # 根据 fps 构建ScliDict
    X = SliceDict()
    for name, path in zip(features, fps):
        data = dataset(path)
        X[name] = data.data.astype(np.float32)
        y = data.label

    return X, y
