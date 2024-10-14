import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import math


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt




class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss




class Bilinear_Pooling(nn.Module):
    def __init__(self,  **kwargs):
        super(Bilinear_Pooling, self).__init__()

    def forward(self, feature_map1, feature_map2):
        feature_map1 = feature_map1.permute(0, 2, 1)
        feature_map2 = feature_map2.permute(0, 2, 1)

        B, C, N1 = feature_map1.size()  # torch.Size([64, 768, 147])
        B, C, N2 = feature_map2.size()

        X = torch.bmm(feature_map1, feature_map2.transpose(1, 2)) / N1
        # torch.Size([64, 768*768])
        X = torch.reshape(X, (B, C * C))
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        bilinear_features = 100 * torch.nn.functional.normalize(X)
        return bilinear_features




class EuclideanLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, x, y):
        B, _ = x.shape
        #margin = 3.0

        dist = torch.sqrt(((x - y) ** 2).sum())
        dist_loss = (dist / B)

        #euclidean_distance = torch.nn.functional.pairwise_distance(x, y)
        #dist_loss = torch.mean(torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        return dist_loss






'''-----------------------  SDPA  ------------------------'''
def Attn_HW(f_K1, f_Q2):
    #B, 3*WH, C = feature1.shape  # torch.Size([64, 147, 768])
    B, N, C = f_K1.shape
    scale = C ** -0.5

    I = (f_Q2 @ f_K1.transpose(-2, -1)) * scale   #（N2,N1）
    I2 = F.softmax(I, 2)
    I1 = F.softmax(I, 1).permute(0, 2, 1)

    return I1, I2







'''-----------------------  DCS  S  ------------------------'''
def Attn_DCS_S(f_K1, f_Q2):
    #B, 3*WH, C = feature1.shape  # torch.Size([64, 147, 768])
    B, N, C = f_K1.shape

    D = torch.cdist(f_Q2, f_K1, p=2)  # B x N2 x C , B x N1 x C --> B x N2 x N1
    D = D - torch.min(D)
    S = torch.max(D) - D

    S1 = F.normalize(S, p=2, dim=2)
    S1 = torch.sum(S1, dim=1)  # B N1
    cross_map1 = F.softmax(S1, dim=-1)

    S2 = F.normalize(S, p=2, dim=1)
    S2 = torch.sum(S2, dim=2)   #B N2
    cross_map2 = F.softmax(S2, dim=-1)


    return cross_map1, cross_map2





'''-----------------------  QCS  SD  ------------------------'''
def Attn_QCS_SD(QK1, QK2, QK3, QK4, k):
    B, N, C = QK1.shape
    k = k + 1

    D_2x1 = torch.cdist(QK2, QK1, p=2)  # B x N2 x C , B x N1 x C --> B x N2 x N1
    D_2x1 = D_2x1 - torch.min(D_2x1)
    S_2x1 = torch.max(D_2x1) - D_2x1

    D_4x3 = torch.cdist(QK4, QK3, p=2)  # B x N4 x C , B x N3 x C --> B x N4 x N3
    D_4x3 = D_4x3 - torch.min(D_4x3)
    S_4x3 = torch.max(D_4x3) - D_4x3

    D_3x1 = torch.cdist(QK3, QK1, p=2)  # B x N3 x N1
    D_4x2 = torch.cdist(QK4, QK2, p=2)  # B x N4 x N2
    D_3x1 = D_3x1 - torch.min(D_3x1)
    D_4x2 = D_4x2 - torch.min(D_4x2)

    ######################## K1 ########################
    S_2x1_n1 = F.normalize(S_2x1, p=2, dim=2)
    S1 = torch.sum(S_2x1_n1, dim=1)  # B x N1
    ''''''''''''''''''''''''''
    D_3x1_n1 = F.normalize(D_3x1, p=2, dim=2)
    D1 = torch.sum(D_3x1_n1, dim=1)  # B x N1

    ''''''''''''''''''''''''''
    SD1 = S1 + k * D1
    cross_map1 = F.softmax(SD1, dim=-1)

    ######################## Q4 ########################
    S_4x3_n4 = F.normalize(S_4x3, p=2, dim=1)
    S4 = torch.sum(S_4x3_n4, dim=2)  # B x N4
    ''''''''''''''''''''''''''
    D_4x2_n4 = F.normalize(D_4x2, p=2, dim=1)
    D4 = torch.sum(D_4x2_n4, dim=2)  # B x N4

    ''''''''''''''''''''''''''
    SD4 = S4 + k * D4
    cross_map4 = F.softmax(SD4, dim=-1)

    ######################## K2 ########################
    S_1x2 = S_2x1.transpose(1, 2)  # B x N1 x N2
    S_1x2_n2 = F.normalize(S_1x2, p=2, dim=2)
    S2 = torch.sum(S_1x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    D_4x2_n2 = F.normalize(D_4x2, p=2, dim=2)
    D2 = torch.sum(D_4x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    SD2 = S2 + k * D2
    cross_map2 = F.softmax(SD2, dim=-1)

    ######################## Q3 ########################
    S_3x4 = S_4x3.transpose(1, 2)  # B x N3 x N4
    S_3x4_n3 = F.normalize(S_3x4, p=2, dim=1)
    S3 = torch.sum(S_3x4_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    D_3x1_n3 = F.normalize(D_3x1, p=2, dim=1)
    D3 = torch.sum(D_3x1_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    SD3 = S3 + k * D3
    cross_map3 = F.softmax(SD3, dim=-1)

    return cross_map1, cross_map2, cross_map3, cross_map4



'''-----------------------   QCS S  ------------------------'''
def Attn_QCS_S(QK1, QK2, QK3, QK4, k):
    B, N, C = QK1.shape
    #k = k + 1

    D_2x1 = torch.cdist(QK2, QK1, p=2)  # B x N2 x C , B x N1 x C --> B x N2 x N1
    D_2x1 = D_2x1 - torch.min(D_2x1)
    S_2x1 = torch.max(D_2x1) - D_2x1

    D_4x3 = torch.cdist(QK4, QK3, p=2)  # B x N4 x C , B x N3 x C --> B x N4 x N3
    D_4x3 = D_4x3 - torch.min(D_4x3)
    S_4x3 = torch.max(D_4x3) - D_4x3

    #D_3x1 = torch.cdist(QK3, QK1, p=2)  # B x N3 x N1
    #D_4x2 = torch.cdist(QK4, QK2, p=2)  # B x N4 x N2
    #D_3x1 = D_3x1 - torch.min(D_3x1)
    #D_4x2 = D_4x2 - torch.min(D_4x2)

    ######################## K1 ########################
    S_2x1_n1 = F.normalize(S_2x1, p=2, dim=2)
    S1 = torch.sum(S_2x1_n1, dim=1)  # B x N1
    ''''''''''''''''''''''''''
    #D_3x1_n1 = F.normalize(D_3x1, p=2, dim=2)
    #D1 = torch.sum(D_3x1_n1, dim=1)  # B x N1

    ''''''''''''''''''''''''''
    #SD1 = S1 + k * D1
    cross_map1 = F.softmax(S1, dim=-1)

    ######################## Q4 ########################
    S_4x3_n4 = F.normalize(S_4x3, p=2, dim=1)
    S4 = torch.sum(S_4x3_n4, dim=2)  # B x N4
    ''''''''''''''''''''''''''
    #D_4x2_n4 = F.normalize(D_4x2, p=2, dim=1)
    #D4 = torch.sum(D_4x2_n4, dim=2)  # B x N4

    ''''''''''''''''''''''''''
    #SD4 = S4 + k * D4
    cross_map4 = F.softmax(S4, dim=-1)

    ######################## K2 ########################
    S_1x2 = S_2x1.transpose(1, 2)  # B x N1 x N2
    S_1x2_n2 = F.normalize(S_1x2, p=2, dim=2)
    S2 = torch.sum(S_1x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    #D_4x2_n2 = F.normalize(D_4x2, p=2, dim=2)
    #D2 = torch.sum(D_4x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    #SD2 = S2 + k * D2
    cross_map2 = F.softmax(S2, dim=-1)

    ######################## Q3 ########################
    S_3x4 = S_4x3.transpose(1, 2)  # B x N3 x N4
    S_3x4_n3 = F.normalize(S_3x4, p=2, dim=1)
    S3 = torch.sum(S_3x4_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    #D_3x1_n3 = F.normalize(D_3x1, p=2, dim=1)
    #D3 = torch.sum(D_3x1_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    #SD3 = S3 + k * D3
    cross_map3 = F.softmax(S3, dim=-1)

    return cross_map1, cross_map2, cross_map3, cross_map4




'''----------------------- QCS D  ------------------------'''
def Attn_QCS_D(QK1, QK2, QK3, QK4, k):
    B, N, C = QK1.shape
    #k = k + 1

    #D_2x1 = torch.cdist(QK2, QK1, p=2)  # B x N2 x C , B x N1 x C --> B x N2 x N1
    #D_2x1 = D_2x1 - torch.min(D_2x1)
    #S_2x1 = torch.max(D_2x1) - D_2x1

    #D_4x3 = torch.cdist(QK4, QK3, p=2)  # B x N4 x C , B x N3 x C --> B x N4 x N3
    #D_4x3 = D_4x3 - torch.min(D_4x3)
    #S_4x3 = torch.max(D_4x3) - D_4x3

    D_3x1 = torch.cdist(QK3, QK1, p=2)  # B x N3 x N1
    D_4x2 = torch.cdist(QK4, QK2, p=2)  # B x N4 x N2
    D_3x1 = D_3x1 - torch.min(D_3x1)
    D_4x2 = D_4x2 - torch.min(D_4x2)

    ######################## K1 ########################
    #S_2x1_n1 = F.normalize(S_2x1, p=2, dim=2)
    #S1 = torch.sum(S_2x1_n1, dim=1)  # B x N1
    ''''''''''''''''''''''''''
    D_3x1_n1 = F.normalize(D_3x1, p=2, dim=2)
    D1 = torch.sum(D_3x1_n1, dim=1)  # B x N1

    ''''''''''''''''''''''''''
    #SD1 = S1 + k * D1
    cross_map1 = F.softmax(D1, dim=-1)

    ######################## Q4 ########################
    #S_4x3_n4 = F.normalize(S_4x3, p=2, dim=1)
    #S4 = torch.sum(S_4x3_n4, dim=2)  # B x N4
    ''''''''''''''''''''''''''
    D_4x2_n4 = F.normalize(D_4x2, p=2, dim=1)
    D4 = torch.sum(D_4x2_n4, dim=2)  # B x N4

    ''''''''''''''''''''''''''
    #SD4 = S4 + k * D4
    cross_map4 = F.softmax(D4, dim=-1)

    ######################## K2 ########################
    #S_1x2 = S_2x1.transpose(1, 2)  # B x N1 x N2
    #S_1x2_n2 = F.normalize(S_1x2, p=2, dim=2)
    #S2 = torch.sum(S_1x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    D_4x2_n2 = F.normalize(D_4x2, p=2, dim=2)
    D2 = torch.sum(D_4x2_n2, dim=1)  # B x N2
    ''''''''''''''''''''''''''
    #SD2 = S2 + k * D2
    cross_map2 = F.softmax(D2, dim=-1)

    ######################## Q3 ########################
    #S_3x4 = S_4x3.transpose(1, 2)  # B x N3 x N4
    #S_3x4_n3 = F.normalize(S_3x4, p=2, dim=1)
    #S3 = torch.sum(S_3x4_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    D_3x1_n3 = F.normalize(D_3x1, p=2, dim=1)
    D3 = torch.sum(D_3x1_n3, dim=2)  # B x N3
    ''''''''''''''''''''''''''
    #SD3 = S3 + k * D3
    cross_map3 = F.softmax(D3, dim=-1)

    return cross_map1, cross_map2, cross_map3, cross_map4









