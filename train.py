from collections import OrderedDict
import numpy as np
import yaml
import pandas as pd
import os
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.dataset import Dataset
import torch
from skorch.callbacks import Callback
from model.MTMF import MTMFTransformer, FT_Vote
from model.FT_transformer import FTTransformer
from utils import setup_seed, evaluate, check_record
from dateset import load_single_features, load_full_features
from model.MBT import MBT


def save_best_model(net, output_dir: str, ):
    """save the best dev model"""
    epoch = len(net.history)
    params_path = output_dir + f"/model_best.pkl"
    optim_path = output_dir + f"/optim_best.pkl"
    history_path = output_dir + f"/history_best.json"
    net.save_params(f_params=params_path,
                    f_optimizer=optim_path,
                    f_history=history_path)


class SaveModel(Callback):
    def __init__(self, disease: str, seed: int):
        self.output_dir = f"./Checkpoints/{disease}/evaluate/{seed}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # save valid scores
        if net.history[-1, 'valid_acc_best']:
            save_best_model(net, self.output_dir)



def train(disease: str, feature: str, seed: int, model_type: str,
          use_config: bool = False, mode: int = 0,
          use_bottleneck: bool = True, btn_init: str = "embed", use_cross_atn: bool = True, noise: float = 0,
          **kwargs):
    if use_config:
        config_path = f"Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = config[model_type][feature]
    else:
        modelconfig = kwargs

    # Set random seed
    setup_seed(seed)
    feature = feature.split(",")
    # Load feature
    if model_type == "FT":
        x_train, x_test, y_train, y_test = load_single_features(seed=seed, disease=disease, feature=feature)
        inputs_dim = OrderedDict({"f1_input": x_train['f1_input'].shape})  # Dict, shape of each input
    elif disease in ['EW-T2D', 'LC', 'C-T2D', 'IBD', 'Obesity']:
        x_train, x_test, y_train, y_test = load_full_features(seed=seed, disease=disease, feature=feature, noise=noise)
        inputs_dim = OrderedDict({"f1_input": x_train['f1_input'].shape,
                                  "f2_input": x_train['f2_input'].shape})  # Dict, shape of each input
    else:
        assert 0, disease

    if model_type not in ['MTMFTransformer', 'FT-Vote', 'MBT']:
        # Concat multi-inputs -- FT/FT-Concat
        # Other models retain the SliceDict format
        x_train = np.concatenate(list(x_train.values()), axis=1).astype(np.float32)
        x_test = np.concatenate(list(x_test.values()), axis=1).astype(np.float32)

    # Set model config
    device = "cuda"
    modelconfig['lr'] = float(modelconfig['lr'])
    lr = modelconfig['lr']
    batch_size = int(modelconfig['batch_size'])

    if model_type == "MTMFTransformer":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck
        modelconfig['btn_init'] = btn_init
        modelconfig['use_cross_atn'] = use_cross_atn

        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    elif "FT" in model_type:
        record = OrderedDict(modelconfig)
        modelconfig['last_layer_query_idx'] = [-1]
        modelconfig['d_out'] = 1
        modelconfig['cat_cardinalities'] = None
        if model_type != "FT-Vote":
            modelconfig['n_num_features'] = x_train.shape[1]
        else:
            modelconfig['n_num_features'] = inputs_dim

    elif model_type == "MBT":
        modelconfig['inputs_dim'] = inputs_dim
        modelconfig['use_bottleneck'] = use_bottleneck

        record = OrderedDict(modelconfig)
        record.pop('inputs_dim')
    else:
        assert 0, model_type

    # Record model config
    record['seed'] = seed
    record['mode'] = mode
    record['feature'] = ','.join(feature)

    logdir = f"./results/{disease}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logpath = f"{logdir}/{model_type}.csv"
    if use_config:  # Reproduce the results
        logpath = f"{logdir}/results-{model_type}_res.csv"
    if noise:
        logpath = f"{logdir}/results-{model_type}_noise_{noise}.csv"

    # check record
    print(logpath)
    if not check_record(record, logpath):
        print("paras has trained.")
        return

    modelconfig.pop('lr')
    modelconfig.pop('batch_size')

    # Init model
    print(modelconfig)
    if model_type == "MTMFTransformer":
        model = MTMFTransformer(**modelconfig).cuda()
    elif model_type == "FT-Concat" or model_type == "FT":
        model = FTTransformer.make_default(**modelconfig).cuda()
    elif model_type == "FT-Vote":
        model = FT_Vote(**modelconfig).cuda()
    elif model_type == "MBT":
        model = MBT(**modelconfig).cuda()
    else:
        assert 0

    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss

    net = NeuralNetClassifier(
        model,
        max_epochs=200,
        criterion=criterion,
        lr=lr,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2, random_state=42),
        device=device,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=15),
                   SaveModel(disease, seed),]
    )

    # train
    net.fit(x_train, y_train)

    # 保存模型
    # save_best_model(net, f"./Checkpoints/{disease}/evaluate/{seed}")

    # 加载模型
    net.load_params(f_params=f"./Checkpoints/{disease}/evaluate/{seed}/model_best.pkl",
                    f_optimizer=f"./Checkpoints/{disease}/evaluate/{seed}/optim_best.pkl",
                    f_history=f"./Checkpoints/{disease}/evaluate/{seed}/history_best.json")

    # test
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
