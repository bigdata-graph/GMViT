from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class SVCNN(nn.Module):
    def __init__(self, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__()
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            print(self.net_2)
            return self.net_2(y.view(y.shape[0], -1))

## Transformers
class Mlp(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N,
                                               C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


def group_pooling(final_views, positions, views_score, group_num):
    interval = 1.0 / group_num

    def onebatch_grouping(onebatch_views, onebatch_positions, onebatch_scores):
        viewgroup_onebatch = [[] for i in range(group_num)]
        scoregroup_onebatch = [[] for i in range(group_num)]
        positions_onebatch = [[] for i in range(group_num)]

        for i in range(group_num):
            left = i * interval
            right = (i + 1) * interval
            for j, score in enumerate(onebatch_scores):
                if left <= score < right:
                    viewgroup_onebatch[i].append(onebatch_views[j])
                    scoregroup_onebatch[i].append(score)
                    positions_onebatch[i].append(onebatch_positions[j])
                else:
                    pass
                
        view_group = [torch.max(torch.stack(views, 0), 0)[0] for views in viewgroup_onebatch if len(views) > 0]
        pos_group = [torch.mean(torch.stack(pos, 0), 0) for pos in positions_onebatch if len(pos) > 0]
        view_group = torch.stack(view_group, 0).cuda()
        pos_group = torch.stack(pos_group, 0).cuda()

        return view_group, pos_group

    view_group = []
    pos_group = []
    for (onebatch_views, onebatch_positions, onebatch_scores) in zip(final_views, positions, views_score):
        V, P = onebatch_grouping(onebatch_views, onebatch_positions, onebatch_scores)
        view_group.append(V)
        pos_group.append(P)
    view_group, pos_group = torch.stack(view_group, 0), torch.stack(pos_group, 0)

    return view_group, pos_group


class GMViT(nn.Module):
    def __init__(self, model, cnn_name='resnet18', num_views=20, group_num=12):
        super().__init__()

        self.trans_dim = 512
        self.depth = 6
        self.drop_path_rate = 0.1
        self.cls_dim = 40
        self.num_heads = 8
        
        self.num_views = num_views
        self.group_num = group_num
        
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            # SVCNN
            self.encoder_net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.encoder_net_2 = model.net.fc
        else:
            self.encoder_net_1 = model.net_1
            self.encoder_net_2 = model.net_2
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            pos = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            # 根号3
            phi = np.sqrt(3)
            pos = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
            
        self.pos = torch.tensor(pos, dtype=torch.float32).cuda()

        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos_1 = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos_2 = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed_1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.pos_embed_2 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks_1 = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.blocks_2 = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.attention = nn.Sequential(nn.Linear(self.trans_dim, 1))

        trunc_normal_(self.cls_token_1, std=.02)
        trunc_normal_(self.cls_pos_1, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        trunc_normal_(self.cls_pos_2, std=.02)
        

    def forward(self, img):

        views = self.num_views
        group_input_tokens = self.encoder_net_1(img)
        f_cnn = group_input_tokens
        
        group_input_tokens = group_input_tokens.view((int(img.shape[0] / views), views, -1))
        pos = self.pos.unsqueeze(0).repeat(group_input_tokens.shape[0], 1, 1)

        cls_tokens = self.cls_token_1.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos_1.expand(group_input_tokens.size(0), -1, -1)
        
        pos = self.pos_embed_1(pos)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks_1(x, pos)
        x = self.norm(x)

        X = x[:, 1:]
        f_vit1 = X
        postion = self.pos.unsqueeze(0).repeat(X.shape[0], 1, 1)
        feature_bank = []
        att_bank = []
        f_vit2 = []
        for xx, pos in zip(X, postion):
            xx = xx.unsqueeze(0)
            pos = pos.unsqueeze(0)
            att = torch.sigmoid(self.attention(xx))
            att_bank.append(att.view(1, -1))
            
            group_input_tokens, pos = group_pooling(xx, pos, att, group_num=self.group_num)

            cls_tokens = self.cls_token_2.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos_2.expand(group_input_tokens.size(0), -1, -1)

            pos = self.pos_embed_2(pos)

            y = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            y = self.blocks_2(y, pos)
            y = self.norm(y)

            concat_f = torch.cat([y[:, 0], y[:, 1:].max(1)[0]], dim=-1)
            f_vit2.append(y[:, 1:].max(1)[0].view(1, -1))
            feature_bank.append(concat_f.squeeze())

        f_vit2 = torch.cat(f_vit2, dim=0)
        att_bank = torch.cat(att_bank, dim=0)
        f_global = self.cls_head_finetune[:2](torch.stack(feature_bank, 0))
        pred = self.cls_head_finetune(torch.stack(feature_bank, 0))
        
        return f_cnn.reshape(-1, 512), f_vit1.reshape(-1, 512), att_bank.reshape(-1, views), f_vit2, f_global, pred
