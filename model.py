import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
import copy

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.1)
        init.normal_(m.bias.data, 0.0, 0.1)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Define the ResNet50-based part Model
class ft_net_LPN(nn.Module):
    def __init__(self, block=[2,4], net="resnet50"):
        super(ft_net_LPN, self).__init__()
        self.block = block

        if net == "resnet50":
            base = models.resnet50(pretrained=True)
            self.embed = 2048
        elif net == "resnet101":
            base = models.resnet101(pretrained=False)
            base.load_state_dict(torch.load("resnet101-63fe2227.pth", "cpu"))
            self.embed = 2048
        elif net == "wide_resnet50_2":
            base = models.wide_resnet50_2(pretrained=False)
            base.load_state_dict(torch.load("wide_resnet50_2-95faca4d.pth", "cpu"))
            self.embed = 2048
        elif net == "wide_resnet101_2":
            base = models.wide_resnet101_2(pretrained=False)
            base.load_state_dict(torch.load("wide_resnet101_2-32ee1156.pth", "cpu"))
            self.embed = 2048
        elif net == "regnet_y_1_6gf":
            base = models.regnet_y_1_6gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_y_1_6gf-b11a554e.pth", "cpu"))
            self.embed = 888
        elif net == "regnet_y_3_2gf":
            base = models.regnet_y_3_2gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_y_3_2gf-9180c971.pth", "cpu"))
            self.embed = 1512
        elif net == "regnet_y_8gf":
            base = models.regnet_y_8gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_y_8gf-dc2b1b54.pth", "cpu"))
            self.embed = 2016
        elif net == "regnet_x_1_6gf":
            base = models.regnet_x_1_6gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_x_1_6gf-e3633e7f.pth", "cpu"))
            self.embed = 912
        elif net == "regnet_x_3_2gf":
            base = models.regnet_x_1_6gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_x_3_2gf-f342aeae.pth", "cpu"))
            self.embed = 1008
        elif net == "regnet_x_8gf":
            base = models.regnet_x_8gf(pretrained=False)
            base.load_state_dict(torch.load("regnet_x_8gf-03ceed89.pth", "cpu"))
            self.embed = 1920

        if 'res' in net:
            self.stem = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3

            base.layer4[0].downsample[0].stride = (1, 1)
            base.layer4[0].conv2.stride = (1, 1)
            self.branch_1 = copy.deepcopy(base.layer4)
            self.branch_2 = copy.deepcopy(base.layer4)
        elif net.startswith("regnet"):
            self.stem = base.stem
            self.layer1 = base.trunk_output.block1
            self.layer2 = base.trunk_output.block2
            self.layer3 = base.trunk_output.block3

            base.trunk_output.block4[0].proj[0].stride = (1, 1)
            base.trunk_output.block4[0].f.b[0].stride = (1, 1)
            self.branch_1 = copy.deepcopy(base.trunk_output.block4)
            self.branch_2 = copy.deepcopy(base.trunk_output.block4)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x1 = self.branch_1(x).float()
        x2 = self.branch_2(x).float()

        g_pool_x1, pool_list_x1 = self.get_part_pool(x1, block=self.block[0])
        g_pool_x2, pool_list_x2 = self.get_part_pool(x2, block=self.block[1])

        pool_list = [g_pool_x1, g_pool_x2] + pool_list_x1 + pool_list_x2
        return pool_list

    def get_part_pool(self, x, block=4):
        B, C, H, W = x.shape
        result = []
        pooling = torch.nn.AdaptiveMaxPool2d(1)
        per_h, per_w = H//(2*block), W//(2*block)
        c_h, c_w = H//2, W//2
        
        for i in range(block):
            if block == 1:
                x_curr = x
            else:
                if i < block - 1:
                    x_curr = x[:, :, c_h-(i+1)*per_h:c_h+(i+1)*per_h, c_w-(i+1)*per_w:c_w+(i+1)*per_w]
                    if i > 0:
                        x_pre = x[:, :, c_h-i*per_h:c_h+i*per_h, c_w-i*per_w:c_w+i*per_w] 
                        x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                        x_curr = x_curr - x_pad
                else:
                    x_pre = x[:, :, c_h-i*per_h:c_h+i*per_h, c_w-i*per_w:c_w+i*per_w]
                    pad_h = c_h-i*per_h
                    pad_w = c_w-i*per_w
                    if x_pre.shape[2]+2*pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H-(x_pre.shape[2]+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep, pad_h, pad_w+ep, pad_w), "constant", 0)
                    x_curr = x - x_pad
            result.append(pooling(x_curr))
        return torch.nn.AdaptiveAvgPool2d(1)(x), result


class three_view_net(nn.Module):
    def __init__(self, class_num, block='2,4'):
        super(three_view_net, self).__init__()
        self.block = [int(i) for i in block.split(',')]

        self.model = ft_net_LPN(block=self.block, net="regnet_y_1_6gf")

        self.feature_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(sum(self.block)+2):
            feature = nn.Linear(self.model.embed, 1024, bias=False)
            feature.apply(weights_init_kaiming)
            bn = nn.BatchNorm1d(1024)
            bn.apply(weights_init_kaiming)
            if i > 1:
                bn.bias.requires_grad_(False)
            self.feature_list.append(feature)
            self.bn_list.append(bn)

        self.classifier_list = nn.ModuleList()
        for _ in range(sum(self.block)+2):
            classifier = nn.Linear(1024, class_num)
            classifier.apply(weights_init_classifier)
            self.classifier_list.append(classifier)

    def forward(self, x):
        pool_list = self.model(x)

        feat_list = []
        bn_list = []
        predict_list = []
        for i in range(sum(self.block)+2):
            pool = pool_list[i].flatten(1)
            feat = self.feature_list[i](pool)
            bn = self.bn_list[i](feat)
            predict = self.classifier_list[i](bn)

            feat_list.append(feat)
            bn_list.append(bn)
            predict_list.append(predict)

        if self.training:
            return predict_list, feat_list
        return torch.stack(bn_list, dim=2)

