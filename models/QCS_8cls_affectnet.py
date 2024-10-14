import torch
import torch.nn as nn
from torch.nn import functional as F
from .ir50 import Backbone
from .vit_model_8cls import VisionTransformer, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from thop import profile
from utils import *


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class CrossAttention(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim * 2)  #
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0)

        self.theta = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x_a, x_p, x_n, x_n2):
        B, N, C = x_a.shape

        x_a = self.norm1(x_a)
        x_p = self.norm1(x_p)
        x_n = self.norm1(x_n)
        x_n2 = self.norm1(x_n2)

        qkv_a = self.proj(x_a).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK1, V1 = qkv_a[0], qkv_a[1]

        qkv_p = self.proj(x_p).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK2, V2 = qkv_p[0], qkv_p[1]

        qkv_n = self.proj(x_n).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK3, V3 = qkv_n[0], qkv_n[1]

        qkv_n2 = self.proj(x_n2).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK4, V4 = qkv_n2[0], qkv_n2[1]

        ##############################################

        k = torch.tanh(self.theta)
        cross_map1, cross_map2, cross_map3, cross_map4 = Attn_QCS_SD(QK1, QK2, QK3, QK4, k) # # B WH WH  torch.Size([64, 49, 49])

        ############################################
        cross_map1 = torch.reshape(cross_map1, shape=(B, N, 1))
        attn_a = cross_map1 * V1  # B N C # torch.Size([64, 147, 768])
        x_a = x_a + attn_a

        cross_map2 = torch.reshape(cross_map2, shape=(B, N, 1))
        attn_p = cross_map2 * V2  # B N C
        x_p = x_p + attn_p

        cross_map3 = torch.reshape(cross_map3, shape=(B, N, 1))
        attn_n = cross_map3 * V3  # B N C
        x_n = x_n + attn_n

        cross_map4 = torch.reshape(cross_map4, shape=(B, N, 1))
        attn_n2 = cross_map4 * V4  # B N C
        x_n2 = x_n2 + attn_n2

        x_a = self.norm2(x_a)
        mlp_a = self.mlp(x_a)
        x_a = x_a + mlp_a

        x_p = self.norm2(x_p)
        mlp_p = self.mlp(x_p)
        x_p = x_p + mlp_p

        x_n = self.norm2(x_n)
        mlp_n = self.mlp(x_n)
        x_n = x_n + mlp_n

        x_n2 = self.norm2(x_n2)
        mlp_n2 = self.mlp(x_n2)
        x_n2 = x_n2 + mlp_n2

        return x_a, x_p, x_n, x_n2, k

class Fusion(nn.Module):

    def __init__(self, num_classes=8):
        super().__init__()

        self.bilinear_pooling = Bilinear_Pooling()
        self.classifier = nn.Linear(768*768, num_classes)

    def forward(self, x_a, x_p, attn_a, attn_p):
        pooling_a = torch.flatten(self.bilinear_pooling(x_a, attn_a), 1)
        out_a = self.classifier(pooling_a)

        pooling_b = torch.flatten(self.bilinear_pooling(x_p, attn_p), 1)
        out_p = self.classifier(pooling_b)

        return out_a, out_p





class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=8, dims=[64, 128, 256], embed_dim=768):
        super().__init__()

        self.img_size = img_size

        self.num_classes = num_classes

        self.VIT_base = VisionTransformer(depth=2, drop_ratio=0.2, embed_dim=embed_dim)
        self.VIT_cross = VisionTransformer(depth=1, drop_ratio=0.5, embed_dim=embed_dim)

        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load(r'.\models\pretrain\ir50.pth', map_location=lambda storage, loc: storage)

        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)

        self.embed_q = nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        self.embed_k = nn.Sequential(nn.Conv2d(dims[1], 768, kernel_size=3, stride=2, padding=1))
        self.embed_v = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)

        self.cross_attention_1 = CrossAttention(embed_dim)
        self.cross_attention_2 = CrossAttention(embed_dim)
        self.cross_attention_3 = CrossAttention(embed_dim)


        #self.fusion = Fusion(num_classes)


    def forward(self, x_a, x_p, x_n, x_n2):

        '''----------------- anchor ----------------'''
        x_a_ir1, x_a_ir2, x_a_ir3 = self.ir_back(x_a)
        x_a_ir1, x_a_ir2, x_a_ir3 = self.conv1(x_a_ir1), self.conv2(x_a_ir2), self.conv3(x_a_ir3)
        #torch.Size([64, 64, 28, 28]) torch.Size([64, 128, 14, 14]) torch.Size([64, 256, 7, 7])
        x_a_o1, x_a_o2, x_a_o3 = self.embed_q(x_a_ir1).flatten(2).transpose(1, 2), self.embed_k(x_a_ir2).flatten(2).transpose(1, 2), self.embed_v(x_a_ir3)
        #torch.Size([64, 49, 768]) torch.Size([64, 49, 768]) torch.Size([64, 49, 768])
        x_a_o = torch.cat([x_a_o1, x_a_o2, x_a_o3], dim=1)
        x_a, x_a_0 = self.VIT_base(x_a_o)

        if x_p == None:
            return x_a

        '''----------------- positive ----------------'''
        x_p_ir1, x_p_ir2, x_p_ir3 = self.ir_back(x_p)
        x_p_ir1, x_p_ir2, x_p_ir3 = self.conv1(x_p_ir1), self.conv2(x_p_ir2), self.conv3(x_p_ir3)
        x_p_o1, x_p_o2, x_p_o3 = self.embed_q(x_p_ir1).flatten(2).transpose(1, 2), self.embed_k(x_p_ir2).flatten(2).transpose(1, 2), self.embed_v(x_p_ir3)
        x_p_o = torch.cat([x_p_o1, x_p_o2, x_p_o3], dim=1)
        x_p, x_p_0 = self.VIT_base(x_p_o)

        '''----------------- negative ----------------'''
        x_n_ir1, x_n_ir2, x_n_ir3 = self.ir_back(x_n)
        x_n_ir1, x_n_ir2, x_n_ir3 = self.conv1(x_n_ir1), self.conv2(x_n_ir2), self.conv3(x_n_ir3)
        x_n_o1, x_n_o2, x_n_o3 = self.embed_q(x_n_ir1).flatten(2).transpose(1, 2), self.embed_k(x_n_ir2).flatten(2).transpose(1, 2), self.embed_v(x_n_ir3)
        x_n_o = torch.cat([x_n_o1, x_n_o2, x_n_o3], dim=1)
        x_n, x_n_0 = self.VIT_base(x_n_o)

        '''----------------- negative2 ----------------'''
        x_n2_ir1, x_n2_ir2, x_n2_ir3 = self.ir_back(x_n2)
        x_n2_ir1, x_n2_ir2, x_n2_ir3 = self.conv1(x_n2_ir1), self.conv2(x_n2_ir2), self.conv3(x_n2_ir3)
        x_n2_o1, x_n2_o2, x_n2_o3 = self.embed_q(x_n2_ir1).flatten(2).transpose(1, 2), self.embed_k(x_n2_ir2).flatten(2).transpose(1, 2), self.embed_v(x_n2_ir3)
        x_n2_o = torch.cat([x_n2_o1, x_n2_o2, x_n2_o3], dim=1)
        x_n2, x_n2_0 = self.VIT_base(x_n2_o)


        '''----------------- attention ----------------'''
        _, N, _ = x_a_o1.shape
        x_a1_1, x_a1_2, x_a1_3 = torch.split(x_a_0, [N, N, N], dim=1)
        x_p1_1, x_p1_2, x_p1_3 = torch.split(x_p_0, [N, N, N], dim=1)
        x_n1_1, x_n1_2, x_n1_3 = torch.split(x_n_0, [N, N, N], dim=1)
        x_n2_1, x_n2_2, x_n2_3 = torch.split(x_n2_0, [N, N, N], dim=1)

        attn_a1, attn_p1, attn_n_1, attn_n2_1, k1 = self.cross_attention_1(x_a1_1, x_p1_1, x_n1_1, x_n2_1)
        attn_a2, attn_p2, attn_n_2, attn_n2_2, k2 = self.cross_attention_2(x_a1_2, x_p1_2, x_n1_2, x_n2_2)
        attn_a3, attn_p3, attn_n_3, attn_n2_3, k3 = self.cross_attention_3(x_a1_3, x_p1_3, x_n1_3, x_n2_3)

        attn_a_o = torch.cat([attn_a1, attn_a2, attn_a3], dim=1)
        attn_p_o = torch.cat([attn_p1, attn_p2, attn_p3], dim=1)
        attn_n_o = torch.cat([attn_n_1, attn_n_2, attn_n_3], dim=1)
        attn_n2_o = torch.cat([attn_n2_1, attn_n2_2, attn_n2_3], dim=1)

        '''----------------- connection ----------------'''
        x_a_0 = x_a_0 + attn_a_o
        x_p_0 = x_p_0 + attn_p_o
        x_n_0 = x_n_0 + attn_n_o
        x_n2_0 = x_n2_0 + attn_n2_o
        out_a, _ = self.VIT_cross(x_a_0)
        out_p, _ = self.VIT_cross(x_p_0)
        out_n, _ = self.VIT_cross(x_n_0)
        out_n2, _ = self.VIT_cross(x_n2_0)

        #out_a, out_p, out_n, out_n2 = self.fusion(x_a_0, x_p_0, x_n_0, x_n2_0, attn_a_o, attn_p_o, attn_n_o, attn_n2_o)


        return x_a, x_p, x_n, x_n2, out_a, out_p, out_n, out_n2, k1,k2,k3



def compute_param_flop():
    model = pyramid_trans_expr2()
    img = torch.rand(size=(1,3,224,224))
    flops, params = profile(model, inputs=(img,))
    print(f'flops:{flops/1000**3}G,params:{params/1000**2}M')


