import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points. This step is analogous to convolution filter act on a local neigbourhood.  
    src^T * dst = xn * xm + yn * ym ；
    sum(src^2, dim=-1) = xn*xn + yn*yn ;
    sum(dst^2, dim=-1) = xm*xm + ym*ym ;

    dist = (xn - xm)^2 + (yn - ym)^2 
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) 
    return dist # (B , N , M)

def index_points(input_points, idx):
    """
    This function creates the data for corresponding index. idx is a proper handle for implementation of PointNet++, but for each idx, we eventually want all the channel information associated with it.
    Noted that the output dimsion would be the same as input points dimension.
    
    - if the input_points are xyz, then the dimension is [B, N, D]
    - if the input_points are points, then the dimension is [B, N, C]

    index_points function are going to be called twice in later functions in order to generate data with dimension [B, N, D+C]
          
    In our code, xyz carry the D information, points carry the C information.

    Input:
        points: input points data, [B, N, C or D depending on the dimension of input]
        idx: sample index data, [B, npoint]
    Return:
        new_points:, indexed points data, [B, npoint, C or D depending on the dimention of input]
    """
    device = input_points.device
    B = input_points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = input_points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    This function is called during sampling layer of set absctraction:Given input points{x1,x2,...,xn}, we use iterative farthest point sampling (FPS)to choose a subset of points{xi1,xi2,...,xim}, such that xijis the most distant point (in metricdistance) from the set{xi1,xi2,...,xij−1}with regard to the rest points.
    
    Input:
        xyz: pointcloud data, [B, N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, D = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #[B, npoint]
    distance = torch.ones(B, N).to(device) * 1e10 #[B, N]
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) 
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  #[0, 1, 2, ..., B -1] [B, 1]
    for point_index in range(npoint):
        centroids[:, point_index] = farthest 
        centroid = xyz[batch_indices, farthest, :].view(B, 1, D)
        dist = torch.sum((xyz - centroid) ** 2, -1) 
        mask = dist < distance #[B, N]
        distance[mask] = dist[mask] 
        farthest = torch.max(distance, -1)[1]

    return centroids #[B, npoint]


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query finds all points that are within a radius to the query point (an upper limit of K is set inimplementation). An alternative range query is K nearest neighbor (kNN) search which finds a fixed number of neighboring points. Compared with kNN, ball query’s local neighborhood guaranteesa fixed region scale thus making local region feature more generalizable across space, which is preferred for tasks requiring local pattern recognition.
    
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, D]
        new_xyz: query points, [B, npoint, D]

    Return:
        index_of_points_in_each_group: grouped points index, [B, npoint, nsample]
    """
    device = xyz.device
    B, N, D = xyz.shape
    _, npoint, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, npoint, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N # mark all the points whose sqrdists is larger than radius^2
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # sort in ascending order along the last dimension and only take nsamples
    group_first = group_idx[:, :, 0].view(B, npoint, 1).repeat([1, 1, nsample]) #[B, npoint, nsample]
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx #[B, npoint, nsample]

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: number of "sampled centroid points"
        radius: radius of circle for local region/group
        nsample: maximum "number of neighborhood points" of centroid point in each local region/group
        xyz: input points position data, [B, N, D]
        points: input points data, [B, N, C]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, D]
        new_points: sampled points data, [B, npoint, nsample, D+C]
    """

    B, N, D = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) # [B, npoints, D]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B, npoints, nsample]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, D]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, D)#[B, npoint, nsample, D]

    if points is not None:
        grouped_points = index_points(points, idx) #points has dimention [B, N, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm #[B, npoint, nsample, C+D = D]
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points # [B, npoint, nsample, D] and [B, npoint, nsample, D+C] respectively


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, D]
        points: input points data, [B, N, C]
    Return:
        new_xyz: sampled points position data, [B, 1, D]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, D = xyz.shape
    new_xyz = torch.zeros(B, 1, D).to(device)
    grouped_xyz = xyz.view(B, 1, N, D)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    '''
    For the detailed description of PointNetSetAbstraction, please refer to Figure 2 of the paper and section 3.2 Hierarchical Point Set Feature Learning
    
    The set abstraction level is made of three key layers:
    - Sampling layer: The Sampling layer selects a set of points from input points, which defines the centroids of local regions
    - Grouping layer: Grouping layer then constructs local region sets by finding “neighboring” points around the centroids
    - PointNet layer: PointNet layeruses a mini-PointNet to encode local region patterns into feature vectors
    '''
    def __init__(self, npoint, radius, nsample, in_channel, mlp, conv_group, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint # Number of "sampled centroid points", npoint
        self.radius = radius # the radius of the sample
        self.nsample = nsample # Maximum "number of neighborhood points" of centroid point in each local region/group
        self.mlp_convs = nn.ModuleList() # predefined convolutional layers
        self.mlp_bns = nn.ModuleList() # predefined batch normalizations
        last_channel = in_channel #how many channels(features) are there to input to the network
        for i in range(len(mlp)):
            out_channel = mlp[i]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, groups=conv_group[i]))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        The Objective of the set abstraction layer is to "condense local features". This is accomplished via the max pooling step of the PointNet
        Pay attention to the channel dimension changes. C -> C_new

        Input:
            xyz: input points position data, [B, D, N]
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, D, npoint]
            new_points_concat: sample points feature data, [B, C_new, npoint]
        """
        xyz = xyz.permute(0, 2, 1) #from [B, D, N] to [B, N, D]
        if points is not None: # there might be dataset where coordinate information is all there is to it. For those dataset, points == None
            points = points.permute(0, 2, 1) #from [B, C, N] to [B, N, C]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)# new_xyz: [B, npoint, nsample D] new_points: [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points))) #[B, C_new, nsample, npoint], C_new is as specified by the last layer of network
        new_points = torch.max(new_points, 2)[0] #[B, C_new, npoint], of the nsample points, only take the maximum value of each element
        new_xyz = new_xyz.permute(0, 2, 1) #[B, D, npoint]
        return new_xyz, new_points #[B, D, npoint] and [B, C_new, npoint] respectively


class PointNetFeaturePropagation(nn.Module):
    '''
    For classification, PointNetFeatureAbstraction is all you need. Yet for point-wise classification, i.e. semantic segmentation, we need to extrapolate
        1. Interpolatation. From sampled centroid points to recovered original input points, the interpolatation part taks 3 points among all the sampled centroid points which have smallest distance to each of original input points to recover this original input point
        2. Skip link concatenation.
        3. Go through unit PointNet

    For detailed description of Feature Propogation, please refer to Figure 2 and section 3.4 of the original paper. 
    '''
    def __init__(self, in_channel, mlp, conv_group):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel # the size of input channel C_less+C_more
        for i in range(len(mlp)):
            out_channel = mlp[i]
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, groups=conv_group[i]))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz_more, xyz_less, points_more, points_less):
        """
        Input:
            xyz_more: the layer with more points with position data [B, D, npoint_more]
            xyz_less: the layer with less points, with position data [B, D, npoint_less]
            points_more: the layer with more points with feature data [B, C_less, npoint_more]
            points_less: the layer with less points with feature data [B, C_more, npoint_less]
        Return:
            new_points: upsampled points data, [B, C_new, npoint_more]
        """
        xyz_more = xyz_more.permute(0, 2, 1) # from [B, D, npoint_more] to [B, npoint_more, D]
        xyz_less = xyz_less.permute(0, 2, 1) # from [B, D, npoint_less] to [B, npoint_less, D]

        points_less = points_less.permute(0, 2, 1) #from [B, C_more, npoint_less] to [B, npoint_less, C_more]
        B, npoint_more, D = xyz_more.shape
        _, npoint_less, _ = xyz_less.shape

        if npoint_less == 1: #if there are only 1 center point left in the layer with less points, then repeat for npoint_more times
            interpolated_points = points_less.repeat(1, npoint_more, 1) #[B, npoint_more, C_more]
        else:
            dists = square_distance(xyz_more, xyz_less) #[B, npoint_more, npoint_less]
            dists, idx = dists.sort(dim=-1) #sort along the last dimension
            dists, idx = dists[:, :, :D], idx[:, :, :D] 
            dists[dists < 1e-5] = 1e-5 #clump the value
            weight = 1.0 / dists  # the size of weight is [B, npoint_more, D]
            weight = weight / torch.sum(weight, dim=-1).view(B, npoint_more, 1)
            interpolated_points = torch.sum(index_points(
                points_less, idx) * weight.view(B, npoint_more, D, 1), dim=2)
        if points_more is not None: 
            points_more = points_more.permute(0, 2, 1) #from [B, C_less, npoint_more] to [B, npoint_more, C_less]
            new_points = torch.cat([points_more, interpolated_points], dim=-1) #[B, npoint_more, C_less+C_more]
        else:
            new_points = interpolated_points #[B, npoint_more, C_more]
        new_points = new_points.permute(0, 2, 1) #from [B, npoint_more, C_more+C_less] to [B, C_more+C_less, npoint_more]
        for i, conv in enumerate(self.mlp_convs): # input dimension = C_more+C_less
            bn = self.mlp_bns[i]
            """ new_points = F.relu(bn(conv(new_points))) #[B, C_new, npoint_more] C_new is speficied by the last layer of network """            
            new_points = conv(new_points)
            new_points = bn(new_points)
            new_points = F.relu(new_points)
        return new_points #[B, C_new, npoint_more]


'''
    self attention
'''
class self_attention(nn.Module):
    def __init__(self, C):
        super(self_attention, self).__init__()
        self.conv_q = nn.Conv1d(C, C//2, 1)
        self.conv_k = nn.Conv1d(C, C//2, 1)
        self.conv_v = nn.Conv1d(C, C, 1)

    def forward(self, x):  # x = [B, C, N]
        Q = self.conv_q(x)  # Q = [B, C/2, N(Q)]
        K = self.conv_k(x)  # K = [B, C/2, N(K)]
        A = Q.permute(0, 2, 1).matmul(K) / math.sqrt(K.shape[1])  # Q*K^T/sqrt(d_k) A=[B,N(Q),N(K)]
        A = A.softmax(dim=-1)
        V = self.conv_v(x)  # V = [B, C, N(K)]
        out = V.matmul(A.permute(0, 2, 1))  # x = [B, C, N(Q)]
        return x + out


'''
    gMLP相关
    代码参考：https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/fightingcv/mlp/g_mlp.py
    论文参考：Pay Attention to MLPs
'''
class gmlp_block(nn.Module):
    def __init__(self, C, d_ffn, N, tiny_attn=False, group_conv=False):  # C=x.shape[2](特征维度), N=x.shape[1](点的数量), d_ffn为中间层维度
        super(gmlp_block, self).__init__()
        assert d_ffn % 2 == 0  # d_ffn为偶数
        self.ln = nn.LayerNorm(C)
        if group_conv==True:
            self.conv1 = nn.Conv1d(C, d_ffn, 1, groups=C//4)
            self.conv2 = nn.Conv1d(int(d_ffn/2), C, 1, groups=C//4)
        else:
            self.conv1 = nn.Conv1d(C, d_ffn, 1)
            self.conv2 = nn.Conv1d(int(d_ffn/2), C, 1)
        self.SGU = spatial_gating_unit(C, d_ffn, N, tiny_attn, group_conv)

    def forward(self, x):  # x为feature map，dim=[B,N,C]
        shortcut = x
        x = self.ln(x)  # 沿channel维度作normalization
        x_attn = x  # [B, N, C]
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.conv1(x)  # [B, d_ffn, N]
        x = F.gelu(x)
        x = x.permute(0, 2, 1)  # [B, N, d_ffn]
        x = self.SGU(x, x_attn)  # [B, N, d_ffn/2]；若tiny_attn为False，则x_attn不起作用
        x = x.permute(0, 2, 1)  # [B, d_ffn/2, N]
        x = self.conv2(x)  # [B, C, N]
        x = x.permute(0, 2, 1)  # [B, N, C]
        return x + shortcut  # [B, N, C]


class spatial_gating_unit(nn.Module):
    def __init__(self, C, d_ffn, N, tiny_attn=False, group_conv=False):
        super(spatial_gating_unit, self).__init__()
        self.ln = nn.LayerNorm(int(d_ffn/2))
        if group_conv==True:
            self.conv = nn.Conv1d(N, N, 1, groups=N//4)
        else:
            self.conv = nn.Conv1d(N, N, 1)
        self.conv.weight = torch.nn.Parameter(torch.randn(self.conv.weight.shape) * 0.01)  # 初始化weight为接近0的小数
        self.conv.bias = torch.nn.Parameter(torch.ones(self.conv.bias.shape))  # 初始化bias为1

        self.tiny_attn = tiny_attn
        if tiny_attn == True:
            self.tiny_attention = tiny_attention(C, d_ffn)

    def forward(self, x, x_attn):  # x=[B,N,d_ffn],x_attn=[B,N,C]
        split = int(x.shape[2] / 2)  # 分割点(d_ffn/2)
        u = x[:, :, :split]
        v = x[:, :, split:]  # 将x沿channel维度分成两部分，每部分为[B, N, d_ffn/2]
        v = self.ln(v)
        v = self.conv(v)  # [B, N, d_ffn/2]
        if self.tiny_attn == True:
            v = v + self.tiny_attention(x_attn)  # [B, N, d_ffn/2]
        return u * v  # [B, N, d_ffn/2]

class tiny_attention(nn.Module):
    def __init__(self, C, d_ffn, d_attn=64):
        super(tiny_attention, self).__init__()
        self.d_attn = d_attn
        self.conv1 = nn.Conv1d(C, 3 * d_attn, 1)
        self.conv2 = nn.Conv1d(d_attn, int(d_ffn/2), 1)

    def forward(self, x):  # x=[B,N,C]
        x = x.permute(0, 2, 1)  # x=[B,C,N]
        qkv = self.conv1(x)  # qkv=[B,3*d_attn,N]
        qkv = qkv.permute(0, 2, 1)  # qkv=[B,N,3*d_attn]
        q = qkv[:, :, :self.d_attn]
        k = qkv[:, :, self.d_attn:2 * self.d_attn]
        v = qkv[:, :, 2 * self.d_attn:]  # 沿channel维度分成三部分（K、Q、V），每部分为[B, N, d_attn]
        w = torch.einsum("bnd,bmd->bnm", q, k)  # w = [B,N,N]
        a = F.softmax(w / math.sqrt(self.d_attn), dim=-1)  # a = [B,N,N]
        x = torch.einsum("bnm,bmd->bnd", a, v)  # x = [B,N,d_attn]
        x = x.permute(0, 2, 1)  # x=[B,d_attn,N]
        x = self.conv2(x)  # x = [B,d_ffn/2,N]
        x = x.permute(0, 2, 1)  # x=[B,N,d_ffn/2]
        return x

'''
    代码参考：https://zhuanlan.zhihu.com/p/374329074
    论文参考：Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks
'''
class External_Attention(nn.Module):
    def __init__(self, C, M):  # M为中间维度
        super(External_Attention, self).__init__()
        self.M_k = nn.Conv1d(C, M, 1, bias=False)  # [B,C,N]->[B,M,N]
        self.M_v = nn.Conv1d(M, C, 1, bias=False)  # [B,M,N]->[B,C,N]

    def forward(self, x):  # x=[B,C,N]
        attn = self.M_k(x)  # attn=[B,M,N]
        attn = F.softmax(attn, dim=2)  # attn=[B,M,N]
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # l1_normalization
        out = self.M_v(attn)  # out=[B,C,N]
        return x + out