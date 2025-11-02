from collections import OrderedDict
import os

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import os
import yaml
import pandas as pd
import torch
import shap
from evaluate import get_trained_model
from utils import *
from dateset import load_full_features


def explain(disease: str, model: str, feature: str, seed: int, device: str):
    # 加载模型参数
    config_path = f"Config/{disease}.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        modelconfig = config[model][feature]

    # 加载数据
    setup_seed(seed)
    feature = feature.split(",")

    # Load feature
    x_train, x_test, y_train, y_test = load_full_features(seed=seed, disease=disease, feature=feature)
    inputs_dim = OrderedDict({"f1_input": x_train['f1_input'].shape,
                              "f2_input": x_train['f2_input'].shape})  # Dict, shape of each input

    # 拼接输入
    x_train = np.concatenate(list(x_train.values()), axis=1).astype(np.float32)
    x_test = np.concatenate(list(x_test.values()), axis=1).astype(np.float32)
    x_train = torch.Tensor(x_train).to(device)
    x_test = torch.Tensor(x_test).to(device)
    x_combined = torch.cat((x_train, x_test), dim=0)  # dim=0 表示在 batch 维度上拼接

    ko_feat = list(pd.read_csv(f"./Data/{disease}/ko_abundance.csv").columns)[2:]
    species_feat = list(pd.read_csv(f"./Data/{disease}/species_abundance.csv").columns)[2:]
    feat_names = ko_feat + species_feat
    # 去除过长featname
    feat_names = [x if len(x) < 20 else x[x.rfind('|') + 1:] for x in feat_names]

    modelconfig['inputs_dim'] = inputs_dim
    modelconfig['use_bottleneck'] = True
    modelconfig['btn_init'] = "embed"
    modelconfig['use_cross_atn'] = True

    net = get_trained_model(disease=disease, seed=seed, modelconfig=modelconfig, explainable=True)

    # 分批计算 SHAP 值
    batch_size = 16  # 你可以根据显存大小调整 batch_size
    num_batches = (x_test.shape[0] + batch_size - 1) // batch_size  # 计算批次数

    shap_values_all = []  # 用于存储所有的 SHAP 值

    print("Calculating SHAP values...")
    # 按批次计算 SHAP 值

    # for i in range(num_batches):
    #     print(f"Batch {i + 1}/{num_batches}")
    #     batch_start = i * batch_size
    #     batch_end = min((i + 1) * batch_size, x_test.shape[0])
    #     x_batch = x_test[batch_start:batch_end]
    #
    #     # 计算当前批次的 SHAP 值
    #     explainer = shap.DeepExplainer(net.module_, x_batch)
    #     shap_values_batch = explainer.shap_values(x_batch, check_additivity=False)
    #
    #     # 将当前批次的 SHAP 值添加到结果中
    #     shap_values_all.append(shap_values_batch)
    # # 拼接所有的 SHAP 值
    # shap_values_combined = np.concatenate(shap_values_all, axis=0)
    # print(shap_values_combined.shape)

    explainer = shap.DeepExplainer(net.module_.to(device), x_test)
    shap_values = explainer.shap_values(x_test, check_additivity=False)

    shap.summary_plot(shap_values.squeeze(), x_test, feature_names=feat_names, rng=np.random.default_rng(), show=False)

    if not os.path.exists(f"./explain/{disease}/{model}"):
        os.makedirs(f"./explain/{disease}/{model}")

    plt.savefig(f"./explain/{disease}/{model}/{seed}.png", dpi=720)

    shap_values = shap_values.squeeze()
    df = pd.DataFrame(shap_values)

    path = f"./explain/{disease}/{model}/{seed}.csv"
    df.to_csv(path, index=False)


