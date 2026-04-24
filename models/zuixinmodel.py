import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import GAT3D

BATCH_NORM_DECAY = 1 - 0.9
BATCH_NORM_EPSILON = 1e-5

def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()


class BalancedVesselAttention(nn.Module):
    """平衡的血管注意力 - 解决过度增强问题"""
    
    def __init__(self, channels):
        super(BalancedVesselAttention, self).__init__()
        self.channels = channels
        
        # 简化的方向检测器 - 减少复杂度
        self.directional_convs = nn.ModuleList([
            # 主要方向
            nn.Conv2d(channels, channels // 4, kernel_size=(1, 3), padding=(0, 1)),  # 水平
            nn.Conv2d(channels, channels // 4, kernel_size=(3, 1), padding=(1, 0)),  # 垂直
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),            # 对角线
            nn.Conv2d(channels, channels // 4, kernel_size=1),                       # 点状
        ])
        
        # 简化的连续性增强器
        self.continuity_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 降低增强强度
        self.enhance_strength = nn.Parameter(torch.tensor(0.2))  # 从0.6降低到0.2
        
    def forward(self, x):
        # 多方向特征提取
        directional_features = []
        for conv in self.directional_convs:
            directional_features.append(conv(x))
        
        # 融合方向性特征
        fused_directional = torch.cat(directional_features, dim=1)
        
        # 简化连续性增强
        vessel_response = self.continuity_enhancer(fused_directional)
        
        # 温和增强输出
        enhanced_x = x + x * vessel_response * self.enhance_strength
        
        return enhanced_x


class SimplifiedSmallVesselBooster(nn.Module):
    """简化的小血管增强器"""
    
    def __init__(self, channels):
        super(SimplifiedSmallVesselBooster, self).__init__()
        self.channels = channels
        
        # 简化的多尺度检测器
        self.detail_detectors = nn.ModuleList([
            nn.Conv2d(channels, channels // 3, 1),           # 点状
            nn.Conv2d(channels, channels // 3, 3, padding=1), # 小区域
            nn.Conv2d(channels, channels // 3, 5, padding=2), # 中等区域
        ])
        
        # 简化的特征聚合器
        self.feature_aggregator = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 降低增强因子
        self.boost_factor = nn.Parameter(torch.tensor(0.25))  # 从0.7降低到0.25
        
    def forward(self, x):
        # 多尺度检测
        detail_features = []
        for detector in self.detail_detectors:
            detail_features.append(detector(x))
        
        # 特征聚合
        aggregated = torch.cat(detail_features, dim=1)
        enhancement_map = self.feature_aggregator(aggregated)
        
        # 温和增强
        boosted_x = x + x * enhancement_map * self.boost_factor
        
        return boosted_x


class SimplifiedFSA(nn.Module):
    """简化的FSA - 减少计算复杂度"""
    
    def __init__(self, channels):
        super(SimplifiedFSA, self).__init__()
        self.channels = channels
        
        # 简化的频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 1),
            nn.Sigmoid()
        )
        
        # 简化的空域分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 可学习的平衡权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 频域分支
        freq_attention = self.freq_branch(x)
        freq_out = x * freq_attention
        
        # 空域分支
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = self.spatial_branch(spatial_input)
        spatial_out = x * spatial_attention
        
        # 动态平衡融合
        alpha_norm = torch.sigmoid(self.alpha)
        beta_norm = torch.sigmoid(self.beta)
        total_weight = alpha_norm + beta_norm + 1e-8
        alpha_norm = alpha_norm / total_weight
        beta_norm = beta_norm / total_weight
        
        combined = freq_out * alpha_norm + spatial_out * beta_norm
        
        # 温和增强
        output = x + (combined - x) * 0.1  # 大幅降低增强强度
        
        return output


class OptimizedMPCA(nn.Module):
    """优化的MPCA - 减少参数和计算量"""
    
    def __init__(self, channels, reduction=8, num_scales=3):  # 减少尺度数量，增加压缩率
        super(OptimizedMPCA, self).__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.reduction = reduction
        
        # 减少尺度的特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1 if i == 0 else 3, 
                         padding=0 if i == 0 else i, dilation=i if i > 0 else 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_scales)
        ])
        
        # 简化的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_scales, max(channels // reduction, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 16), channels, 1),
            nn.Sigmoid()
        )
        
        # 简化的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 降低输出权重
        self.output_weight = nn.Parameter(torch.tensor(0.08))  # 从0.15降低到0.08
        
    def forward(self, x):
        # 多尺度特征提取
        scale_features = []
        for conv in self.multi_scale_convs:
            feat = conv(x)
            scale_features.append(feat)
        
        # 归一化权重
        normalized_weights = F.softmax(self.scale_weights, dim=0)
        
        # 加权融合
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weighted_feat = feat * normalized_weights[i]
            weighted_features.append(weighted_feat)
        
        # 特征融合
        fused_features = torch.cat(weighted_features, dim=1)
        
        # 通道注意力
        ca_weight = self.channel_attention(fused_features)
        attended_features = fused_features * ca_weight.expand_as(fused_features)
        
        output = self.fusion_conv(attended_features)
        
        # 残差连接
        return x + output * self.output_weight


class OptimizedResBlockWithPool(nn.Module):
    """优化的ResBlock - 解决注意力冲突"""
    
    def __init__(self, in_channels, out_channels):
        super(OptimizedResBlockWithPool, self).__init__()
        self.same_channels = in_channels == out_channels

        # 主要卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 快捷连接
        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 简化注意力模块 - 只保留最有效的两个
        self.vessel_attention = BalancedVesselAttention(out_channels)  # 专注血管检测
        self.spatial_attention = SimplifiedSpatialAttention()         # 空间注意力
        
        # 简化的权重参数
        self.vessel_weight = nn.Parameter(torch.tensor(0.3))   # 降低权重避免过拟合
        self.spatial_weight = nn.Parameter(torch.tensor(0.2))  # 适中的空间权重
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x if self.same_channels else self.shortcut(x)

        # 第一个卷积块
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.dropout(out1)  # 添加正则化

        # 第二个卷积块
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.relu2(out2)

        # 简化的注意力处理
        vessel_enhanced = self.vessel_attention(out2)
        spatial_enhanced = self.spatial_attention(out1)
        
        # 平衡融合，避免梯度冲突
        out = (out2 + 
               vessel_enhanced * self.vessel_weight + 
               spatial_enhanced * self.spatial_weight)
        
        out = self.pool(out)
        return out


class OptimizedResBlock(nn.Module):
    """优化的ResBlock（无池化）"""
    
    def __init__(self, in_channels, out_channels):
        super(OptimizedResBlock, self).__init__()
        self.same_channels = in_channels == out_channels

        # 主要卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接
        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU(inplace=True)
        
        # 简化注意力模块
        self.vessel_attention = BalancedVesselAttention(out_channels)
        self.spatial_attention = SimplifiedSpatialAttention()
        
        # 简化权重
        self.vessel_weight = nn.Parameter(torch.tensor(0.25))
        self.spatial_weight = nn.Parameter(torch.tensor(0.15))
        
        # 添加dropout
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x if self.same_channels else self.shortcut(x)

        # 第一个卷积块
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.dropout(out1)

        # 第二个卷积块
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 += identity
        out2 = self.relu2(out2)

        # 简化的注意力处理
        vessel_enhanced = self.vessel_attention(out2)
        spatial_enhanced = self.spatial_attention(out1)
        
        # 平衡融合
        out = (out2 + 
               vessel_enhanced * self.vessel_weight + 
               spatial_enhanced * self.spatial_weight)
        
        return out


# 保持原有的基础模块
class SimplifiedSpatialAttention(nn.Module):
    """简化的空间注意力"""
    
    def __init__(self, kernel_size=7):
        super(SimplifiedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class AdaptiveTripletAttention(nn.Module):
    """自适应三元注意力 - 优化版"""
    
    def __init__(self, no_spatial=False):
        super(AdaptiveTripletAttention, self).__init__()
        self.no_spatial = no_spatial
        
        self.cw_attn = SimplifiedSpatialAttention()
        self.hc_attn = SimplifiedSpatialAttention()
        
        if not no_spatial:
            self.hw_attn = SimplifiedSpatialAttention()
        
        num_attentions = 2 if no_spatial else 3
        # 初始化为更平衡的权重
        self.attention_weights = nn.Parameter(torch.ones(num_attentions) * 0.3)

    def forward(self, x):
        attentions = []
        
        # 通道-宽度注意力
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw_attn(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()
        attentions.append(x_out1)
        
        # 高度-通道注意力
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc_attn(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()
        attentions.append(x_out2)
        
        if not self.no_spatial:
            # 高度-宽度注意力
            x_out3 = self.hw_attn(x)
            attentions.append(x_out3)
        
        # 归一化权重后加权融合
        normalized_weights = F.softmax(self.attention_weights, dim=0)
        output = torch.zeros_like(x)
        for i, attn in enumerate(attentions):
            output += attn * normalized_weights[i]
        
        # 添加残差连接，降低变化幅度
        output = x + (output - x) * 0.2
            
        return output


# 保持其他所有基础模块不变...
class BNReLU(nn.Module):
    def __init__(self, out_channels, activation='relu', nonlinearity=True, init_zero=False):
        super(BNReLU, self).__init__()
        self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_DECAY, eps=BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act = get_act(activation)
        else:
            self.act = None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out = self.norm(input)
        if self.act is not None:
            out = self.act(out)
        return out


# 保持其他所有基础模块不变...
class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class GroupPointWise(nn.Module):
    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(torch.Tensor(in_channels, heads, proj_channels // heads))
        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).float()
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):
    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)
        o = self.self_attention(q=q, k=k, v=v)
        return o


class BotBlock(nn.Module):
    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=None):
        super(BotBlock, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size=3, padding=1, stride=stride),
                BNReLU(target_dimension, activation=activation, nonlinearity=True),
            )
        else:
            self.shortcut = None

        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3, padding=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3, padding=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )
        self.last_act = get_act(activation)

    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        Q_h = Q_w = 4
        N, C, H, W = x.shape
        P_h, P_w = H // Q_h, W // Q_w

        x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)
        out = self.conv1(x)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)
        out = self.conv2(out)
        out = self.conv3(out)

        N1, C1, H1, W1 = out.shape
        out_re = out.reshape(N, C1, H, W)
        out = out_re
        out += shortcut
        out = self.last_act(out)

        return out


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def _make_bot_layer(ch_in, ch_out):
    W = H = 4
    dim_in = ch_in
    dim_out = ch_out

    stage5 = []
    stage5.append(
        BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=1, target_dimension=dim_out)
    )

    return nn.Sequential(*stage5)


# 使用优化版本
ResBlockWithPool = OptimizedResBlockWithPool
ResBlock = OptimizedResBlock
TripletAttention = AdaptiveTripletAttention


class OptimizedGNN_UNet(nn.Module):
    """优化的GNN U-Net - 解决注意力冲突问题"""
    
    def __init__(self, img_ch=3, output_ch=1):
        super(OptimizedGNN_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = _make_bot_layer(ch_in=img_ch, ch_out=64)
        self.Conv2 = OptimizedResBlockWithPool(64, 128)
        self.Conv3 = OptimizedResBlockWithPool(128, 256)
        self.Conv4 = OptimizedResBlockWithPool(256, 512)
        self.Conv5 = OptimizedResBlockWithPool(512, 1024)
        self.TA = AdaptiveTripletAttention()

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.att5 = Attention_Gate(512)
        self.Up_conv5 = OptimizedResBlock(1024, 512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.att4 = Attention_Gate(256)
        self.Up_conv4 = OptimizedResBlock(512, 256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.att3 = Attention_Gate(128)
        self.Up_conv3 = OptimizedResBlock(256, 128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.att2 = Attention_Gate(64)
        self.Up_conv2 = OptimizedResBlock(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
        self.gnn = GAT3D(in_channels=1024, out_channels=1024, num_layers=1, heads=2)
        self.bn_gnn = nn.BatchNorm2d(1024)
        self.endpoint_head = nn.Conv2d(64, 1, kernel_size=1)
        self.path_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # 简化的血管检测头
        self.small_vessel_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 简化的连续性增强头
        self.continuity_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 可学习的融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 0.3, 0.2]))  # 主输出，小血管，连续性
        
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)

        B, C, H, W = x5.shape
        x5_reshaped = x5.view(B, C, -1).permute(0, 2, 1)
        x5_gnn = self.gnn(x5_reshaped)
        x5_gnn = x5_gnn.permute(0, 2, 1).view(B, C, H, W)
        x5 = self.bn_gnn(x5_gnn + x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x5 = self.TA(x5)
        x4 = self.TA(x4)

        x_4 = self.att5(x4, self.pool_reverse(x5))
        d5 = torch.cat((x_4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.TA(x3)
        x_3 = self.att4(x3, self.pool_reverse(x4))
        d4 = torch.cat((x_3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.TA(x2)
        x_2 = self.att3(x2, self.pool_reverse(x3))
        d3 = torch.cat((x_2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.TA(x1)
        x_1 = self.att2(x1, self.pool_reverse(x2))
        d2 = torch.cat((x_1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 主要输出
        d1 = self.Conv_1x1(d2)
        
        # 辅助输出
        small_vessel_pred = self.small_vessel_head(d2)
        continuity_pred = self.continuity_head(d2)
        endpoint_pred = self.endpoint_head(d2)
        path_pred = self.path_head(d2)
        
        # 动态加权融合 - 使用可学习权重
        weights = F.softmax(self.fusion_weights, dim=0)
        enhanced_pred = (d1 * weights[0] + 
                        small_vessel_pred * weights[1] + 
                        continuity_pred * weights[2])

        return enhanced_pred, endpoint_pred, path_pred

    def pool_reverse(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class Attention_Gate(nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        feat1 = F.relu(self.conv1(x1))
        feat2 = F.relu(self.conv3(x2))
        concat_feat = torch.cat([feat1, feat2], dim=1)
        attention_weights = self.sigmoid(self.conv2(concat_feat))
        weighted_x1 = x1 * attention_weights
        return weighted_x1


# 为了向后兼容，保留原始类名
GNN_UNet = OptimizedGNN_UNet