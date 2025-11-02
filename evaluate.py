import os
from collections import OrderedDict

import pandas as pd
import torch
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit

from dateset import load_full_features
import yaml
from utils import setup_seed, evaluate

from model.MTMF import MTMFTransformer
from model.MSFT_explainable import MTMFTransformer_explainable


def get_trained_model(disease: str, seed: int, modelconfig: dict, explainable: bool = False):
    lr = float(modelconfig['lr'])
    batch_size = int(modelconfig['batch_size'])
    modelconfig.pop('lr')
    modelconfig.pop('batch_size')

    if not explainable:
        model = MTMFTransformer(**modelconfig).to("cuda")
    else:
        model = MTMFTransformer_explainable(**modelconfig).to("cuda")

    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss

    # 创建网络
    net = NeuralNetClassifier(
        model,
        max_epochs=200,
        criterion=criterion,
        lr=lr,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2, random_state=42),
        device="cuda",
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        batch_size=batch_size,
    )

    # 加载模型
    net.initialize()
    # 加载模型
    savepath = f"./Checkpoints/{disease}/evaluate/{seed}"
    net.load_params(f_params=f"{savepath}/model_best.pkl",
                    f_optimizer=f"{savepath}/optim_best.pkl",
                    f_history=f"{savepath}/history_best.json")

    return net

def eval(model: str, disease: str, feature: str,  seed: int):
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

    modelconfig['inputs_dim'] = inputs_dim
    modelconfig['use_bottleneck'] = True
    modelconfig['btn_init'] = "embed"
    modelconfig['use_cross_atn'] = True

    record = OrderedDict(modelconfig)
    record.pop('inputs_dim')

    record['seed'] = seed
    record['mode'] = 0
    record['feature'] = ','.join(feature)

    logdir = f"./results/{disease}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logpath = f"{logdir}/{model}_evaluate.csv"

    net = get_trained_model(disease=disease, seed=seed, modelconfig=modelconfig)

    # 评估模型
    scores, df = evaluate(net, x_test, y_test)

    record.update(scores)
    # df.to_csv(f"{logdir}/results-{model_type}_{seed}.csv")  # 保存每次的结果

    try:
        res_df = pd.read_csv(logpath)
        record_df = pd.DataFrame(record, index=[0])
        res_df = pd.concat([res_df, record_df])
    except:
        res_df = pd.DataFrame(record, index=[0])

    res_df.to_csv(logpath, index=False)

