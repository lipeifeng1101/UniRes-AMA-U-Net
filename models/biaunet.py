import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute=True, fold_heads=False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute = absolute
        self.fold_heads = fold_heads

        self.emb_w = nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h = nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits = self.absolute_logits(q)
        if self.absolute:
            logits += abs_logits
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h = self.emb_h[:, None, :]
        emb_w = self.emb_w[None, :, :]
        emb = emb_h + emb_w
        abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


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


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CrossLayerAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossLayerAttention, self).__init__()
        self.fc_query = nn.Conv2d(in_channels, out_channels*2, kernel_size=1)
        self.fc_key = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=1)  # 修改为in_channels
        self.fc_value = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=1)  # 修改为与fc_key相同的输出通道数
        self.fc_out = nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  # 在空间维度上进行softmax

    def forward(self, x_upper, x_lower):
        # 计算query和key
        query = self.fc_query(x_upper)
        key = self.fc_key(x_lower)

        # 计算注意力权重
        # 将query和key展平成二维矩阵进行点积
        query_flat = query.view(query.size(0), -1, query.size(1))  # 注意这里的size(1)是out_channels
        key_flat = key.view(key.size(0), -1, key.size(1))  # 同上
        attention_scores = torch.bmm(query_flat, key_flat.transpose(1, 2)).div(query.size(1) ** 0.5)  # 使用out_channels进行缩放
        attention = self.softmax(attention_scores)

        # 计算value并进行加权求和
        value = self.fc_value(x_lower)
        value_flat = value.view(value.size(0), -1, value.size(1))
        out_flat = torch.bmm(attention, value_flat)
        out = out_flat.view(value.size(0), value.size(1), value.size(2), value.size(3))  # 恢复原始的空间维度

        # 通过一个卷积层来调整输出通道数（可选）
        out = self.fc_out(out)

        # 如果需要将输出与x_lower进行融合，可以在这里进行
        # 确保out和x_lower的通道数是相同的
        out = out + x_lower  # 如果通道数不同，这里会报错，需要调整

        # 由于我们修改了fc_value的输出通道数，这里可能需要重新考虑是否要与x_lower相加
        # 如果相加，可能需要先通过1x1卷积调整x_lower的通道数

        return out

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            #nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def _make_bot_layer(ch_in, ch_out):
    W = H = 4
    dim_in = ch_in
    dim_out = ch_out

    stage5 = []

    stage5.append(
        BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=1, target_dimension=dim_out)
    )

    return nn.Sequential(*stage5)



class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # 上采样的 l 卷积
        x1 = self.W_x(x)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # concat + relu
        psi = self.relu(g1 + x1)          #1x256x64x64di
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)               #得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）
        # 返回加权的 x
        return x * psi                    #与low-level feature相乘，将权重矩阵赋值进去


class GT_U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1,):
        super(GT_U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

        self.Conv1 = _make_bot_layer(ch_in=img_ch, ch_out=64)
        self.Conv2 = _make_bot_layer(ch_in=64, ch_out=128)
        #self.att1 = CrossLayerAttention(64, 64)  # 在第二层后应用注意力机制
        self.Conv3 = _make_bot_layer(ch_in=128, ch_out=256)
        
        #self.att2 = CrossLayerAttention(256, 256)  # 在第二层后应用注意力机制
        self.Conv4 = _make_bot_layer(ch_in=256, ch_out=512)
        #self.att3 = CrossLayerAttention(512, 512)
        self.Conv5 = _make_bot_layer(ch_in=512, ch_out=1024)
        #self.att4 = CrossLayerAttention(1024, 1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = _make_bot_layer(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = _make_bot_layer(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = _make_bot_layer(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = _make_bot_layer(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)               ## 1*3*512*512 ->conv(3,64)->conv(64,64)-> 1*64*512*512

        x2_ = self.Maxpool(x1)            # 1*64*512*512 -> 1*64*256*256
        x2 = self.Conv2(x2_)              # 1*64*256*256 ->conv(64,128)->conv(128,128)-> 1*128*256*256
        #x2 = self.att1(x2_, x2)

        x3 = self.Maxpool(x2)            # 1*128*256*256 -> 1*128*128*128
        x3 = self.Conv3(x3)              ## 1*128*128*128 ->conv(128,256)->conv(256,256)->  1*256*128*128

        x4 = self.Maxpool(x3)            # 1*256*128*128 -> 1*256*64*64
        x4 = self.Conv4(x4)              ## 1*256*64*64 ->conv(256,512)->conv(512,512)-> 1*512*64*64

        x5 = self.Maxpool(x4)            ## 1*512*64*64 -> 1*512*32*32
        x5 = self.Conv5(x5)             ## 1*512*32*32->conv(512,1024)->conv(1024,1024)-> 1*1024*32*32

        # decoding + concat path
        d5 = self.Up5(x5)                ## 1*1024*32*32 ->Upsample-> 1*1024*64*64 -> conv(1024,512) ->1*512*64*64
        x4 = self.Att5(g=d5, x=x4)        ## 2(1*512*64*64) -> 1*1*64*64 ->1*512*64*64
        d5 = torch.cat((x4, d5), dim=1)    ## 1*1024*64*64
        d5 = self.Up_conv5(d5)              ## 1*1024*64*64 ->conv(1024,512)->conv(512,512)-> 1*512*64*64

        d4 = self.Up4(d5)                   #1*512*64*64->Upsample-> 1*512*128*128 -> conv(512,256) ->1*256*128*128
        x3 = self.Att4(g=d4, x=x3)          ## 2(1*256*128*128) -> 1*1*128*128 ->1*256*128*128
        d4 = torch.cat((x3, d4), dim=1)     ## 1*512*128*128
        d4 = self.Up_conv4(d4)              ## 1*512*128*128 ->conv(512,256) -> conv(512,512)-> 1*256*128*128

        d3 = self.Up3(d4)                   #1*256*128*128->Upsample-> 1*256*256*256 -> conv(256,128) ->1*128*256*256
        x2 = self.Att3(g=d3, x=x2)          #2(1*128*256*256) -> 1*1*256*256 -> 1*128*256*256
        d3 = torch.cat((x2, d3), dim=1)       #1*256*256*256
        d3 = self.Up_conv3(d3)                #1*256*256*256->conv(256,128) -> conv(128,128)-> 1*128*256*256

        d2 = self.Up2(d3)                      #1*128*256*256->Upsample-> 1*128*512*512 -> conv(128,64) ->1*64*512*512
        x1 = self.Att2(g=d2, x=x1)              #2(1*64*512*512)  -> 1*1*512*512  ->1*64*512*512
        d2 = torch.cat((x1, d2), dim=1)         #1*128*512*512
        d2 = self.Up_conv2(d2)                  #1*128*512*512 -> conv(128,64) ->1*64*512*512 -> conv(64,64) ->1*64*512*512

        d1 = self.Conv_1x1(d2)                  #1*64*512*512 -> 1*3*512*512
        #d1 = self.sigmoid(d1)

        return d1