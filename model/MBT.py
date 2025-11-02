import torch
from typing import Dict, List
from collections import OrderedDict
from model.FT_transformer import NumericalFeatureTokenizer, CLSToken, FTTransformer
from model.MTMF import TransformerBlock
from torch import nn
import numpy as np


class Bottleneck(nn.Module):
    def __init__(self, d_token: int, f_names: List):
        super().__init__()
        btn_block_config = FTTransformer.get_default_transformer_config(n_blocks=1)
        btn_block_config['d_token'] = d_token
        self.btn_blocks = nn.ModuleDict({
            f"{k}_btn_block": TransformerBlock(**btn_block_config)
            for k in f_names
        })

    def forward(self, bottleneck, **features_embed):
        new_features_embed, btn_hats = OrderedDict(), OrderedDict()
        # For each embed
        for name, embed in features_embed.items():
            # print(name, embed.shape, bottleneck.shape)
            # print(bottleneck.shape, embed.shape)
            tmp_btn = torch.cat([embed, bottleneck], dim=1)
            tmp_btn_hat_combined = self.btn_blocks[f"{name}_btn_block"](tmp_btn)
            new_features_embed[name] = tmp_btn_hat_combined[:, :embed.shape[1], :]
            btn_hats[name] = tmp_btn_hat_combined[:, embed.shape[1]:, :]

        #         print(go_btn_hat_combined.shape, go_embed.shape, go_btn_hat.shape)
        # AVG
        btn_attn_avg = torch.mean(torch.stack(list(btn_hats.values()), dim=-1), dim=-1)
        # bottleneck = torch.mean(btn_attn_avg, dim=-1)

        #         print()
        return btn_attn_avg, new_features_embed


class MLP(nn.Module):
    def __init__(self, d_token, hidden_size):
        super().__init__()
        if hidden_size == 0:
            self.out_layer = nn.Sequential(
                nn.Linear(d_token, 1)
            )
            nn.init.zeros_(self.out_layer[0].weight)
        else:
            self.out_layer = nn.Sequential(
                nn.Linear(d_token, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            # nn.init.zeros_(self.out_layer[0].weight)
            nn.init.zeros_(self.out_layer[2].weight)

    def forward(self, x):
        out = self.out_layer(x)
        return out


class MBT(nn.Module):
    def __init__(self, n_layers: int, m_layers: int, num_bottleneck: int, hidden_size: int,
                 use_bottleneck: bool, inputs_dim: Dict):
        super(MBT, self).__init__()
        self.n_layer = n_layers
        self.inputs_dim = inputs_dim
        self.num_bottleneck = num_bottleneck

        self.use_bottleneck = use_bottleneck

        if self.n_layer:
            ft_blocks_config = FTTransformer.get_default_transformer_config(n_blocks=self.n_layer)
            ft_blocks_config['n_blocks'] = 1
            self.d_token = ft_blocks_config['d_token']
        else:
            ft_blocks_config = FTTransformer.get_default_transformer_config(n_blocks=1)
            self.d_token = ft_blocks_config['d_token']

        # Tokenizer
        # 使用 ModuleDict 动态创建 NumericalFeatureTokenizer 实例
        self.feature_tokenizers = nn.ModuleDict({
            f"{k}_Tokenizer": NumericalFeatureTokenizer(n_features=v[1], d_token=self.d_token, bias=True,
                                                        initialization='uniform')
            for k, v in inputs_dim.items()
        })
        self.cls_token = CLSToken(
            self.d_token, 'uniform'
        )

        if self.n_layer:
            ft_blocks_config = FTTransformer.get_default_transformer_config(n_blocks=self.n_layer)
            ft_blocks_config['d_token'] = self.d_token
            self.transformer_blocks = nn.ModuleDict({
                f"{k}_transformer_blocks": TransformerBlock(**ft_blocks_config)
                for k, v in inputs_dim.items()
            })

        # Bottleneck Layer -- n layers for transformer, m layers for multi-transformer
        if self.use_bottleneck:
            self.Bottleneck_layers = nn.ModuleList([Bottleneck(d_token=self.d_token,
                                                               f_names=list(inputs_dim.keys()))
                                                    for _ in range(m_layers)])

        # Representation MLP
        self.mlps = nn.ModuleDict({
            f"{k}_mlp": MLP(d_token=self.d_token, hidden_size=hidden_size)
            for k, v in inputs_dim.items()
        })

    def forward(self, **features):
        # print("============  numpy to tensor  =================\n\n")
        # numpy --> tensor
        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[name] = torch.from_numpy(feature).type(torch.float32).to('cuda')

        # Generate Embedding
        features_embed = OrderedDict()
        # print(" ================================ Generate embedding ==================================")
        for name, feature in features.items():
            features_embed[name] = self.feature_tokenizers[f"{name}_Tokenizer"](feature)
            features_embed[name] = self.cls_token(features_embed[name])  # Add CLS token
            # print(name, feature.shape, "   --->   ", features_embed[name].shape)
        #     print(name, feature.shape, "   --->   ", features_embed[name].shape)
        # print()

        # Bottleneck input --- Init
        if self.use_bottleneck:
            bottleneck = torch.normal(mean=0, std=0.02,
                                      size=(1, self.num_bottleneck, self.d_token)).cuda()
            bottleneck = bottleneck.expand(list(features_embed.values())[0].shape[0], -1, -1)

        # 先进行FT-Transformer-Block  更新 ko_embed, go_embed
        # print(" ================================ Self Attention ==================================")
        if self.n_layer:
            for name, embed in features_embed.items():
                # print(self.transformer_blocks[f"{k}_transformer_blocks"])
                features_embed[name] = self.transformer_blocks[f"{name}_transformer_blocks"](embed)  # 更新embed
            #     print(name, embed.shape, "   --->   ", features_embed[name].shape)
            # print()

        # Bottleneck
        if self.use_bottleneck:
            for idx, layer in enumerate(self.Bottleneck_layers):
                # print(f"Bottleneck_layer --- {idx + 1}")
                bottleneck, features_embed = layer(bottleneck, **features_embed)

        # 提取单个模态的CLS特征
        feature_repr = OrderedDict()
        for name, embed in features_embed.items():
            feature_repr[name] = embed[:, -1]
            # print(tmp_repr.shape)

        out = {}
        x_pool = 0
        for name, feature in feature_repr.items():
            # print(self.FT[f"{name}_FT"])
            out[name] = self.mlps[f"{name}_mlp"](feature)
            x_pool += out[name]

        x_pool /= len(out)
        return x_pool
