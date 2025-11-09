from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from util.transformer import Decoder, Embeddings, MultiHeadedAttention, PositionwiseFeedForward, Generator, DecoderLayer, rate

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            print(f"embed_dim: {embed_dim}")
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        print(f"After patch_embed: {x.shape}")
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
   
        print(f"After transformer blocks: {x.shape}")
        if self.global_pool:
            # x = self.norm(x)      
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            print(f"After global pool: {x.shape}")
            outcome = self.fc_norm(x)
            print(f"outcome: {outcome.shape}")
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
class GATDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_weight, num_heads=4, use_residual=True): 
        super().__init__()
        self.num_labels = out_dim
        self.in_dim = in_dim
        self.label_onehot = torch.eye(out_dim)  # (num_labels, num_labels)
        self.use_residual = use_residual

        # GAT输入维度：label onehot + image feature (768)
        self.input_dim = out_dim + in_dim  # 例如 10 + 768 = 778

        self.gat1 = GATv2Conv(
            in_channels=self.input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False,
            edge_dim=1
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False,
            edge_dim=1
        )

        # 将 image_features 降维 768 → 128
        self.image_proj = nn.Linear(in_dim, 128)

        intermediate_dim = hidden_dim * num_heads // 2  # 可调整
        # GAT输出 + 降维后的 image_features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_heads + 128, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1)
        )

        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(self, image_features):
        # 归一化
        image_features = F.normalize(image_features, p=2, dim=1)  # L2 norm，每个样本单位向量
        B, D = image_features.size()  # D 应该为 768
        label_features = self.label_onehot.to(image_features.device)  # (num_labels, out_dim)
        edge_index = self.edge_index.to(image_features.device)
        edge_weight = self.edge_weight.to(image_features.device)


        outputs = []
        for i in range(B):
            # print("label_features.shape: ", label_features.shape)
            # print("image_features.shape: ", image_features.shape)
            img_feat = image_features[i]  # (768,)

            # 用于 GAT 输入：与 label 拼接
            # img_feat_expand = img_feat.unsqueeze(0).expand(self.num_labels, -1)  # (num_labels, 768)
            img_feat_expand = img_feat.unsqueeze(0).repeat(self.num_labels, 1)

            x = torch.cat([label_features, img_feat_expand], dim=1)  # (num_labels, out_dim + in_dim)
            # print("x.shape: ", x.shape)
            x1 = self.gat1(x, edge_index, edge_attr=edge_weight.unsqueeze(1))  # (num_labels, hidden_dim * heads) 

            x1 = F.relu(x1)
            if self.use_residual and x1.shape == x.shape:
                x1 = x + x1
            # print("x1.shape: ", x1.shape)
            x2 = self.gat2(x1, edge_index, edge_attr=edge_weight.unsqueeze(1))  # (num_labels, hidden_dim * heads) 

            x2 = F.relu(x2)
            
            if self.use_residual and x2.shape == x1.shape:
                x2 = x1 + x2
            # print("x2.shape: ", x2.shape)
            # 降维后的 image_features
            img_feat_proj = F.relu(self.image_proj(img_feat))  # (128,)
            # print("img_feat_proj.shape: ", img_feat_proj.shape)
            # img_feat_proj_expand = img_feat_proj.unsqueeze(0).expand(self.num_labels, -1)  # (num_labels, 128)
            img_feat_proj_expand = img_feat_proj.unsqueeze(0).repeat(self.num_labels, 1)  # (num_labels, 128)

            # 拼接 GAT 输出和降维后的 image_features
            combined = torch.cat([x2, img_feat_proj_expand], dim=1)  # (num_labels, hidden*heads + 128)
            # print("combined.shape: ", combined.shape)
            out = self.final_mlp(combined).squeeze(-1)  # (num_labels,)
            # print("out.shape: ", out.shape)
            # exit()
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (B, num_labels)

class ViTWithGATWrapper(nn.Module):
    def __init__(self, vit_encoder, edge_index, edge_weight, num_labels, in_dim=768, 
                 hidden_dim=64, num_heads=4):  # ViT base 默认是 768
        super().__init__()
        self.vit = vit_encoder
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # 初始化 GATDecoder
        self.decoder = GATDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_labels,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_heads=num_heads
        )

    def forward(self, x):
        x = self.vit.forward_features(x)
        return self.decoder(x)

    @property
    def blocks(self):
        return self.vit.blocks