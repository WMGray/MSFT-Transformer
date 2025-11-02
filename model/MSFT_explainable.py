import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import torch.nn.init as init

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from model.FT_transformer import (MultiheadAttention, CLSToken, FTTransformer,
                                  Transformer, _make_nn_module, NumericalFeatureTokenizer)

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_token: int,
                 n_blocks: int,
                 attention_n_heads: int,
                 attention_dropout: float,
                 attention_initialization: str,
                 attention_normalization: str,
                 ffn_d_hidden: int,
                 ffn_dropout: float,
                 ffn_activation: str,
                 ffn_normalization: str,
                 residual_dropout: float,
                 prenormalization: bool,
                 first_prenormalization: bool,
                 last_layer_query_idx: Union[None, List[int], slice],
                 n_tokens: Optional[int],
                 kv_compression_ratio: Optional[float],
                 kv_compression_sharing: Optional[str],
                 head_activation: ModuleType,
                 head_normalization: ModuleType, ):
        super().__init__()
        self.blocks = nn.ModuleList([])

        def make_kv_compression():
            assert (
                    n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert (
                            kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert (
                x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            #             query_idx = (
            #                 self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            #             )
            query_idx = None
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )

            #             if query_idx is not None:  # 不需要CLS进行预测，不需要提取
            #                 x = x[:, query_idx]

            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)

            #             print("######### self._end_residual(layer, 'ffn', x, x_residual)")
            #             print(x.shape)
            x = layer['output'](x)

        #             print("######### x = layer['output'](x)")
        #             print(x.shape)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, dropout_rate: float):
        super(MLP, self).__init__()
        layers = []
        input_units = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(input_units, units))
            layers.append(nn.GELU())  # Equivalent to tf.nn.gelu in Keras
            layers.append(nn.Dropout(dropout_rate))
            input_units = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Bottleneck(nn.Module):
    def __init__(self, num_bottleneck: int, btn_block_config: Dict, f_names: List,
                 btn_init: str,
                 use_cross_atn: bool):
        super(Bottleneck, self).__init__()
        self.btn_init = btn_init
        self.use_cross_atn = use_cross_atn

        self.d_token = btn_block_config['d_token']
        self.btn_linear = nn.Linear(num_bottleneck, num_bottleneck)
        # 使用正态分布初始化权重，均值为0，标准差为0.02
        self.projection_dim = self.d_token


        self.btn_blocks = nn.ModuleDict({
            f"{k}_btn_block": TransformerBlock(**btn_block_config)
            for k in f_names
        })

        if self.use_cross_atn:
            d_k = d_v = int(self.d_token / 8)
            # 创建交叉注意力
            self.cross_attns = nn.ModuleDict({
                f"{k}_cross": CrossAttention(self.d_token, self.d_token, d_k, d_v, 8)
                for k in f_names
            })

    def forward(self, bottleneck, init_embed, **features_embed):
        if self.btn_init == "embed":
            bottleneck = self.btn_linear(bottleneck)
        # 在第2个维度上添加新的维度
        bottleneck = bottleneck.unsqueeze(2)
        # 在第3个维度上复制projection_dim次
        bottleneck = bottleneck.expand(-1, -1, self.projection_dim)

        new_features_embed, btn_hats = OrderedDict(), OrderedDict()
        """添加交叉注意力"""
        if self.use_cross_atn:
            f1, f2 = list(features_embed.keys())  # feature name
            features_embed[f1] = self.cross_attns[f"{f1}_cross"](features_embed[f1], init_embed[f1])
            features_embed[f2] = self.cross_attns[f"{f2}_cross"](features_embed[f2], init_embed[f2])

        # For each embed
        for name, embed in features_embed.items():
            # print(name, embed.shape, bottleneck.shape)
            tmp_btn = torch.cat([embed, bottleneck], axis=1)
            tmp_btn_hat_combined = self.btn_blocks[f"{name}_btn_block"](tmp_btn)
            new_features_embed[name] = tmp_btn_hat_combined[:, :embed.shape[1], :]
            btn_hats[name] = tmp_btn_hat_combined[:, embed.shape[1]:, :]


        #         print(go_btn_hat_combined.shape, go_embed.shape, go_btn_hat.shape)
        # AVG
        btn_attn_avg = torch.mean(torch.stack(list(btn_hats.values()), dim=-1), dim=-1)
        bottleneck = torch.mean(btn_attn_avg, dim=-1)

        #         print()

        return bottleneck, new_features_embed


class MTMFTransformer_explainable(nn.Module):
    def __init__(self, n_layers: int, num_bottleneck: int,
                 use_bottleneck: bool, btn_init: str, use_cross_atn: bool,
                 inputs_dim: Dict):
        super(MTMFTransformer_explainable, self).__init__()
        self.n_layer = n_layers
        self.inputs_dim = inputs_dim
        self.num_bottleneck = num_bottleneck

        self.use_bottleneck = use_bottleneck
        self.btn_init = btn_init
        self.use_cross_atn = use_cross_atn

        ft_blocks_config = FTTransformer.get_default_transformer_config(n_blocks=self.n_layer)
        ft_blocks_config['n_blocks'] = 1
        self.d_token = ft_blocks_config['d_token']

        # 使用 ModuleDict 动态创建 NumericalFeatureTokenizer 实例
        self.feature_tokenizers = nn.ModuleDict({
            f"{k}_Tokenizer": NumericalFeatureTokenizer(n_features=v[1], d_token=self.d_token, bias=True,
                                                        initialization='uniform')
            for k, v in inputs_dim.items()
        })
        self.cls_token = CLSToken(
            self.d_token, 'uniform'
        )

        # Prep Bottleneck
        if self.btn_init == "embed":
            tmp_len = sum(value[1] + 1 for value in inputs_dim.values())  # +1 -- CLS
            input_1 = tmp_len * self.d_token
            self.prep_linear = nn.Linear(input_1, num_bottleneck)
            # 使用正态分布初始化权重，均值为0，标准差为0.02
            init.normal_(self.prep_linear.weight, mean=0, std=0.02)
            init.constant_(self.prep_linear.bias, 0)  # 初始化偏差为0

        # Bottleneck
        self.Bottleneck_layers = nn.ModuleList([
            Bottleneck(num_bottleneck=num_bottleneck,
                       btn_block_config=ft_blocks_config,
                       f_names=list(inputs_dim.keys()),
                       btn_init=self.btn_init,
                       use_cross_atn=self.use_cross_atn)
            for _ in range(self.n_layer)])

        # Flatten
        self.flatten = nn.Flatten()

        # Outputlayers
        if self.use_bottleneck:
            last_input_size = self.d_token * len(inputs_dim) + num_bottleneck
        else:
            last_input_size = self.d_token * len(inputs_dim)

        self.output = nn.Sequential(
            nn.LayerNorm(last_input_size, eps=1e-5),
            nn.ReLU(),
            nn.Linear(last_input_size, 1)
        )
        # nn.init.zeros_(self.output[2].weight)

    def forward(self, feature):
        # print("============  numpy to tensor  =================\n\n")
        # numpy --> tensor
        # print("============  numpy to tensor dict  =================\n\n")
        features = OrderedDict()
        # 根据self.inputs_dim的key，将numpy转换为tensor
        start = 0
        for k, v in self.inputs_dim.items():
            end = start + v[1]
            features[k] = feature[:, start:end]  # 拆分为多维度特征
            start = end
            # print(k, features[k].shape)

        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[name] = torch.from_numpy(feature).type(torch.float32).to('cuda')

        # Generate Embedding
        features_embed = OrderedDict()
        # print(" ================================ Generate embedding ==================================")
        for name, feature in features.items():
            # print(self.feature_tokenizers.keys())
            features_embed[name] = self.feature_tokenizers[f"{name}_Tokenizer"](feature)
            features_embed[name] = self.cls_token(features_embed[name])  # Add CLS token
            # print(name, feature.shape, "   --->   ", features_embed[name].shape)
        #     print(name, feature.shape, "   --->   ", features_embed[name].shape)
        # print()

        """# Init embedding 用于交叉注意力"""
        init_embed = features_embed

        # prep bottleneck inputs
        # 将所有 embedding 向量拼接在一起
        if self.btn_init == "embed":
            bottleneck_input = torch.cat([features_embed[key].view(features_embed[key].size(0), -1)
                                          for key in features_embed], dim=1)
            # print(bottleneck_input.shape)
            bottleneck = self.prep_linear(bottleneck_input)
            # print(f"Prep bottleneck: {bottleneck.shape}")
        else:
            bottleneck = torch.normal(mean=0, std=0.02,
                                      size=(list(features_embed.values())[0].shape[0],
                                            self.num_bottleneck)).cuda()

        # Bottleneck
        for idx, layer in enumerate(self.Bottleneck_layers):
            # print(f"Bottleneck_layer --- {idx + 1}")
            bottleneck, features_embed = layer(bottleneck, init_embed, **features_embed)

        # 提取单个模态的CLS特征
        feature_repr = OrderedDict()
        for name, embed in features_embed.items():
            feature_repr[name] = embed[:, -1]
            # print(tmp_repr.shape)

        if self.use_bottleneck:
            btn_representation = self.flatten(bottleneck)
            tmp_feats = list(feature_repr.values())
            tmp_feats.insert(1, btn_representation)  # 两个特征时，插入到中间
            all_feats = torch.cat(tmp_feats, dim=1)

        else:
            all_feats = torch.cat(list(feature_repr.values()), dim=1)

        # #         print(f"Features: ko: {ko_fetures.shape} go: {go_fetures.shape} btn: {btn_representation.shape}")
        # #         print(f"ALL Feature Size: {all_feats.shape}")

        """ new trail """
        output = self.output(all_feats)
        return output


