# ---encoding:utf-8---
import math
import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import torch
import torch.nn as nn
import numpy as np
import config
from mamba_ssm import Mamba2
import torch.nn.functional as F

config = config.get_train_config()

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差捷径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.leaky_relu(out, 0.1)


class OptimizedCNN(nn.Module):

    def __init__(self,
                 input_channels=1,
                 layer_channels=None,
                 downsample_steps=None):
        super(OptimizedCNN, self).__init__()

        # 初始层
        if downsample_steps is None:
            downsample_steps = [1, 3]
        if layer_channels is None:
            layer_channels = [8, 16, 32, 64, 128, 256]
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, layer_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(layer_channels[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1)
        )

        # 动态构建残差块
        self.layers = nn.ModuleList()
        for i in range(len(layer_channels) - 1):
            downsample = (i in downsample_steps)  # 是否在该层下采样
            self.layers.append(
                ResidualBlock(layer_channels[i], layer_channels[i + 1], downsample)
            )
            if i % 2 == 1:  # 每隔一个残差块添加Dropout
                self.layers.append(nn.Dropout2d(0.2))

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.prelu = nn.PReLU()

        # 残差连接的快捷路径（Shortcut）
        if in_channels != out_channels:
            # 如果输入和输出通道数不同，使用1x1卷积调整通道
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            # 否则直接使用恒等映射
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)  # 处理快捷路径
        out = self.conv(x)
        out = self.bn(out)
        out += identity  # 残差连接
        out = self.prelu(out)  # 在残差相加后应用激活
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        gap = self.gap(x).view(b, c)
        weights = self.fc(gap)
        return weights.view(b, c, 1, 1)


class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x1, x2):

        channel_weights = self.channel_att(x2)
        x2_weighted = x2 * channel_weights

        x2_pooled = self.gap(x2_weighted).squeeze(-1).squeeze(-1)

        x2_expanded = x2_pooled.unsqueeze(1).expand(-1, x1.size(1), -1)
       
        fused_input = torch.cat([x1, x2_expanded], dim=-1)
        fused = self.mlp(fused_input)

        return fused

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(768, 256)
        self.value = nn.Linear(768, 256)

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)

        # 计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 上下文向量
        context = torch.matmul(attn_weights, V)
        fused = context + x1
        return fused


class OMetaNet(nn.Module):
    def __init__(self, config, num_layers=4, num_heads=4, drop_out=0.2, in_channels1=7):
        super(OMetaNet, self).__init__()

        global max_len, n_layers, n_head, d0_model, d_model, d_ff, d_k, d_v, vocab_size, device

        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d0_model = config.dim_embedding
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        base = num_heads * 8

        conv_output_channels = [base * (2 ** i) for i in range(num_layers)]
        kernel_sizes = [3 + 2 * i for i in range(num_layers)]

        self.conv_layers = self._make_conv_layers(num_layers, in_channels1, conv_output_channels, kernel_sizes)

        self.attention1 = nn.MultiheadAttention(embed_dim=conv_output_channels[-1], num_heads=8)

        self.mycnn = OptimizedCNN(
        input_channels=1,
        layer_channels=[8, 16, 32, 64, 128, 256],
        downsample_steps=[1, 3]
        )

        self.fff = CrossAttentionFusion()

        self.mamba = Mamba2(d_model=768, d_state=64, d_conv=4, expand=4)

        self.fusion = ChannelAttentionFusion()

        self.fc_task = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, x1, x2, x3):

        # x1 deal
        x1 = x1.permute(0, 2, 1)
        x1 = self._apply_conv_layers(self.conv_layers, x1)
        x1 = x1.permute(0, 2, 1)

        identity1 = x1
        x1, _ = self.attention1(x1, x1, x1)
        x1 += identity1

        # x2 deal
        x2 = x2.float()
        x2 = x2.unsqueeze(1)
        x2 = self.mycnn(x2)

        # x3 deal
        x3 = x3 + self.mamba(x3)
        x3 = x3 + self.mamba(x3)
        x3 = x3 + self.mamba(x3)
        x3 = x3 + self.mamba(x3)

        x3 = torch.mean(x3, dim=1)

        # fusion
        fusion = self.fusion(x1, x2)

        fusion = torch.mean(fusion, dim=1)

        a = self.fff(fusion, x3)

        reduction_feature = self.fc_task(a)

        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)

        return logits_clsf, reduction_feature

    def _make_conv_layers(self, num_layers, in_channels, out_channels_list, kernel_sizes):

        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels_list[i], kernel_sizes[i]))
            in_channels = out_channels_list[i]
        return layers

    def _apply_conv_layers(self, layers, x):

        for layer in layers:
            x = layer(x)
        return x


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.fill_diagonal_(0)

        self_mask = torch.eye(labels.size(0), device=device)
        mask = mask * (1 - self_mask)

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss

