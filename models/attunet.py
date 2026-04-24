import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_NORM_DECAY = 1 - 0.9  # pytorch batch norm `momentum = 1 - counterpart` of tensorflow
BATCH_NORM_EPSILON = 1e-5

def get_act(activation):
    """Only supports ReLU and SiLU/Swish.
    获取激活函数"""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

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
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product #{16384,4,4,4,4}
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k) #{16384,4,4,4,4,4}
        if self.relative:
            logits += self.relative_logits(q)#{16384，4，4，4，4，4}
        weights = torch.reshape(logits, [-1, heads, h, w, h * w]) #{16384,4,4,4,16}
        weights = F.softmax(weights, dim=-1)#{16384,4,4,4,16}
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])#{16384,4,4,4,4,4}
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v) #{16384,4,4,4,4}
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim]) #{16384,4,4,16}
        return attn_out

    def relative_logits(self, q): #{q 16384,4,4,4,4}
        # Relative logits in width dimension.  #{rel_logw  16384,4,4,4,4,4}
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension  #{ rel_h   16384,4,4,4,4,4}
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):  #{rel_k 7,4}
        bs, heads, h, w, dim = q.shape  #{bs 16384 heads=4 dim = 4  h = 4 w = 4}
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k) #{16384,4,4,4,7}
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1]) # {16384,16,4,7}
        rel_logits = self.rel_to_abs(rel_logits)#{16384，16，4，4}
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w]) #{16384，4，4，4，4}
        rel_logits = torch.unsqueeze(rel_logits, dim=3) #返回一个新的张量，对输入的既定位置插入维度 1 {16384，4，4，1，4，4}
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1) #函数可以对张量进行重复扩充 {16384，4，4，4，4，4}
        rel_logits = rel_logits.permute(*transpose_mask) # 将维度换位 #{16384，4，4，4，4，4}
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape #{length = 4,bs = 16384,heads = 16,_ = 7}{x 16384,16,4,7}
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda() #返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor {16384,16,4,1}
        x = torch.cat([x, col_pad], dim=3) #{16384,16,4,8}
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()  #整理形状 {16384，16，32}
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda() #{全是0， 16384，16，3}
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)#{16384，16，35}
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1]) #{16384，16，5，7}
        final_x = final_x[:, :, :length, length - 1:] #{16384，16，4，4}
        return final_x



class GroupPointWise(nn.Module):
    """入数据进行通道维度的转置和与参数 self.w 的乘积运算。"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01) #用于对模型参数进行正态分布初始化。标准差设为1

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC 之前的input是{16384，16，4，4}
        input = input.permute(0, 2, 3, 1).float() #{变成了16384，4，4，16}
        """
        将输入数据的维度顺序从 PyTorch 默认的 BCHW（批大小、通道数、高度、宽度）顺序调整为 TensorFlow 默认的 BHWC（批大小、高度、宽度、通道数）顺序 
        .float() 操作是将输入数据的数据类型转换为浮点型
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w) #函数用于进行张量乘法运算
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

    def forward(self, input): # 这里的input是{16384，16，4，4}
        q = self.q_proj(input)  #Q{ 16384,4,4,4,4}
        k = self.k_proj(input)  #K{ 16384,4,4,4,4}
        v = self.v_proj(input)  #V{ 16384,4,4,4,4}

        o = self.self_attention(q=q, k=k, v=v) #O{16384,4,4,16}
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
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # 池化层是一个模块 TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3, padding=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )
        self.last_act = get_act(activation)

    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x) #进来1出去64 先进行卷积在进行数据归一化
        else:
            shortcut = x
        Q_h = Q_w = 4
        N, C, H, W = x.shape
        P_h, P_w = H // Q_h, W // Q_w  # 这里使用了整数除法（//）保证结果为整数  变为16 16

        x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)  # 将输入的张量 x 从四维张量变形为一个二维的张量，对每一个小块进行处理
        # x 变为16384，1，4，4
        out = self.conv1(x) # 包含了一个卷积层和一个激活函数。out变为了16384，16，4，4
        out = self.mhsa(out) #这是多头自注意力 做完了多头自注意力变为了 16384 ，4，4，16
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order 用于交换张量的维度顺序维度顺序为 (N, Q_h, Q_w, dim_out) 将维度顺序改变为 (N, dim_out, Q_h, Q_w)，16384，16，4，4
        out = self.conv2(out) #一个激活函数看stride的大小维度不变
        out = self.conv3(out) # 一个卷积层和一个激活函数 变味了16384.64.4.4

        N1, C1, H1, W1 = out.shape #{ c1 = 64 H1 = 4 N1 = 16384 W1 = 4}
        out_re = out.reshape(N, C1, H, W) #是将 out 进行形状变换（reshape）操作，将其从一个四维张量变换为一个新的四维张量。
        #上面64，64，64，64
        out = out_re
        out += shortcut   #这是相加为维度保持不变
        out = self.last_act(out) # 获 取激活函数

        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),#用于对卷积层的输出进行批标准化操作
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


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out

from skimage.morphology import skeletonize

class GT_U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1,):
        super(GT_U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

        self.Conv1 = _make_bot_layer(ch_in=img_ch, ch_out=64)
        self.Conv2 = _make_bot_layer(ch_in=64, ch_out=128)
        self.Conv3 = _make_bot_layer(ch_in=128, ch_out=256)
        self.Conv4 = _make_bot_layer(ch_in=256, ch_out=512)
        self.Conv5 = _make_bot_layer(ch_in=512, ch_out=1024)
        self.TA = TripletAttention()

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.att5 = Attention_Gate(512)
        self.Up_conv5 = _make_bot_layer(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.att4 = Attention_Gate(256)
        self.Up_conv4 = _make_bot_layer(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.att3 = Attention_Gate(128)
        self.Up_conv3 = _make_bot_layer(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.att2 = Attention_Gate(64)
        self.Up_conv2 = _make_bot_layer(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        #print(x.size())## 1*3*512*512
        x1 = self.Conv1(x)               ## 1*3*512*512 ->conv(3,64)->conv(64,64)-> 1*64*512*512

        x2 = self.Maxpool(x1)            # 1*64*512*512 -> 1*64*256*256
        x2 = self.Conv2(x2)              # 1*64*256*256 ->conv(64,128)->conv(128,128)-> 1*128*256*256


        x3 = self.Maxpool(x2)            # 1*128*256*256 -> 1*128*128*128
        x3 = self.Conv3(x3)              ## 1*128*128*128 ->conv(128,256)->conv(256,256)->  1*256*128*128

        x4 = self.Maxpool(x3)            # 1*256*128*128 -> 1*256*64*64
        x4 = self.Conv4(x4)              ## 1*256*64*64 ->conv(256,512)->conv(512,512)-> 1*512*64*64

        x5 = self.Maxpool(x4)            ## 1*512*64*64 -> 1*512*32*32
        x5 = self.Conv5(x5)             ## 1*512*32*32->conv(512,1024)->conv(1024,1024)-> 1*1024*32*32

        # decoding + concat path
        d5 = self.Up5(x5)  ## 1*1024*32*32 ->Upsample-> 1*1024*64*64 -> conv(1024,512) ->1*512*64*64
        x5 = self.TA(x5)
        x4 = self.TA(x4)

        x_4 = self.att5(x4,self.pool_reverse(x5))  #需要切割x4      ## 2(1*512*64*64) -> 1*1*64*64 ->1*512*64*64
        d5 = torch.cat((x_4, d5), dim=1)    ## 1*1024*64*64
        d5 = self.Up_conv5(d5)              ## 1*1024*64*64 ->conv(1024,512)->conv(512,512)-> 1*512*64*64

        d4 = self.Up4(d5)  # 1*512*64*64->Upsample-> 1*512*128*128 -> conv(512,256) ->1*256*128*128
        x3 = self.TA(x3)
        x_3 = self.att4(x3,self.pool_reverse(x4))          ## 2(1*256*128*128) -> 1*1*128*128 ->1*256*128*128
        d4 = torch.cat((x_3, d4), dim=1)     ## 1*512*128*128
        d4 = self.Up_conv4(d4)              ## 1*512*128*128 ->conv(512,256) -> conv(512,512)-> 1*256*128*128

        d3 = self.Up3(d4)                   #1*256*128*128->Upsample-> 1*256*256*256 -> conv(256,128) ->1*128*256*256
        x2 = self.TA(x2)
        x_2 = self.att3(x2,self.pool_reverse(x3))
        d3 = torch.cat((x_2, d3), dim=1)       #1*256*256*256
        d3 = self.Up_conv3(d3)                #1*256*256*256->conv(256,128) -> conv(128,128)-> 1*128*256*256

        d2 = self.Up2(d3)    #1*128*256*256->Upsample-> 1*128*512*512 -> conv(128,64) ->1*64*512*512
        x1 = self.TA(x1)
        x_1 = self.att2(x1, self.pool_reverse(x2))
        d2 = torch.cat((x_1, d2), dim=1)         #1*128*512*512
        d2 = self.Up_conv2(d2)#1*128*512*512 -> conv(128,64) ->1*64*512*512 -> conv(64,64) ->1*64*512*512

        d1 = self.Conv_1x1(d2)                  #1*64*512*512 -> 1*3*512*512


        return d1,d2




    def pool_reverse(self, x):
        # 简单的上采样函数，实际中应使用 nn.Upsample 或其他上采样方法
        return F.interpolate(x, scale_factor=2, mode='nearest')


        
class Attention_Gate(nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        # 注意力模块的卷积层，用于提取特征
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1 和 x2 分别来自编码层的上下两层
        # 首先对 x1 和 x2 进行卷积操作
        feat1 = F.relu(self.conv1(x1))
        feat2 = F.relu(self.conv3(x2))

        # 将 feat1 和 feat2 进行拼接，然后通过一个卷积层得到注意力权重
        concat_feat = torch.cat([feat1, feat2], dim=1)
        attention_weights = self.sigmoid(self.conv2(concat_feat))

        # 使用注意力权重对 x1 进行加权
        weighted_x1 = x1 * attention_weights

        return weighted_x1