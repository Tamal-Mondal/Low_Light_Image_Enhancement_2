import torch
import torch.nn as nn
import torch.nn.functional as F

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


################################## SELF-ATTENTION MODULE ################################

#Reference/Source: https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bn = False, bias = True)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


################################################# END ###################################################


class enhance_net_nopool(nn.Module):
    def __init__(self, scale_factor):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

        # CBAM layer
        self.cbam_1 = CBAM(gate_channels = 32, reduction_ratio = 16, pool_types = ["avg", "max", "lp", "lse"])
        self.cbam_2 = CBAM(gate_channels = 3, reduction_ratio = 1, pool_types = ["avg", "max", "lp", "lse"])

    def enhance(self, x, x_r, add_extra_itr = False):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)

        # Adding 4 more iterations
        if(add_extra_itr):
            x = x + x_r * (torch.pow(x, 2) - x)
            x = x + x_r * (torch.pow(x, 2) - x)
            x = x + x_r * (torch.pow(x, 2) - x)
            x = x + x_r * (torch.pow(x, 2) - x)

        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return enhance_image

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # print("\n==========Input shape: ", x_down.shape)
        x1 = self.relu(self.cbam_1(self.e_conv1(x_down)))
        x2 = self.relu(self.cbam_1(self.e_conv2(x1)))
        x3 = self.relu(self.cbam_1(self.e_conv3(x2)))
        x4 = self.relu(self.cbam_1(self.e_conv4(x3)))
        x5 = self.relu(self.cbam_1(self.e_conv5(torch.cat([x3, x4], 1))))
        x6 = self.relu(self.cbam_1(self.e_conv6(torch.cat([x2, x5], 1))))
        x_r = torch.tanh(self.cbam_2(self.e_conv7(torch.cat([x1, x6], 1))))
        
        #x1 = self.relu(self.e_conv1(x_down))
        #x2 = self.relu(self.e_conv2(x1))
        #x3 = self.relu(self.e_conv3(x2))
        #x4 = self.relu(self.e_conv4(x3))
        #x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        #x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        #x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        #print("\n==========x_r shape: ", x_r.shape)
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x, x_r, add_extra_itr = False)
        return enhance_image, x_r
