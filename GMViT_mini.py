import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import torchvision.models as models

class Student_CNN(nn.Module):
    def __init__(self):
        super(Student_CNN, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = norm_layer(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = norm_layer(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class GMViT_mini(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.num_views = args.num_views
        self.group_num = args.group_num
        self.encoder_net_1 = model

        self.mlp1 = nn.Linear(512, 512)
        self.attention = nn.Linear(512, 1)
        self.mlp2 = nn.Linear(512, 512)

        self.norm = nn.LayerNorm(512)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, args.num_category)
        )

    def forward(self, x):
        views = self.num_views
        f_cnn = self.encoder_net_1(x)

        x = f_cnn.view((int(f_cnn.shape[0] / views), views, -1))

        x = self.mlp1(x)
        f_vit1 = self.norm(x)

        att = torch.sigmoid(self.attention(f_vit1))
        x = group_pooling(x, att, group_num=self.group_num)

        f_vit2 = []
        for a in x:
            a = a.max(0)[0].view(1, -1)
            f_vit2.append(a)
        f_vit2 = torch.cat(f_vit2, dim=0)

        f_vit2 = self.mlp2(f_vit2)
        f_vit2 = self.norm(f_vit2)

        f_global = self.cls_head_finetune[:2](f_vit2)

        pred = self.cls_head_finetune(f_vit2)

        return f_cnn.reshape(-1, 512), f_vit1.reshape(-1, 512), att.reshape(-1, views), f_vit2, f_global, pred


def group_pooling(final_views, views_score, group_num):
    interval = 1.0 / group_num

    def onebatch_grouping(onebatch_views, onebatch_scores):
        viewgroup_onebatch = [[] for i in range(group_num)]

        for i in range(group_num):
            left = i * interval
            right = (i + 1) * interval
            for j, score in enumerate(onebatch_scores):
                if left <= score < right:
                    viewgroup_onebatch[i].append(onebatch_views[j])
                else:
                    pass

        view_group = [torch.max(torch.stack(views, 0), 0)[0] for views in viewgroup_onebatch if len(views) > 0]  # [G, 512]

        view_group = torch.stack(view_group, 0).cuda()
        return view_group

    view_group = []
    for (onebatch_views, onebatch_scores) in zip(final_views, views_score):
        V = onebatch_grouping(onebatch_views, onebatch_scores)
        view_group.append(V)
    return view_group