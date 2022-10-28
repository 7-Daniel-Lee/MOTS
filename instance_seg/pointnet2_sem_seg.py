'''
PointNet++ for Semantic Segmentation. There are two major components:
1. PointNetSetAbstraction, where sampling takes place
2. PointNetFeaturePropogation, where interpolation takes place to provide the context of semantic segmentation

NOMENCLATURE:
    B is the number of data in each batch
    N is the total number of input points
    C is the dimension of feature or channels
    D is the dimension of coordinate,

    npoint = the number of centers
    nsample = the number of points sampled from each circle 

The dimension of C and D can be extremely confusing, In our code, the D, dimension of the coordinate is 2, x_cc and y_cc and C, the dimension of feature is 2, vr_compensated and rcs.

For more details of these two functions, please refer to Figure 2 of the original paper.
For a detailed explanation concerning parameter choice, please refer to README.md.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from instance_seg.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation, gmlp_block, External_Attention, self_attention

class get_pointnet2_for_semantic_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample, second_layer_npoint,
                 second_layer_radius, second_layer_nsample, group_conv=False):
        super(get_pointnet2_for_semantic_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同

    def forward(self, xyz, dataset_D):
        # The PointNet++ based semantic segmentation network consists two branches: One for outputting predicted semantic information of every point,
        # one for outputting predicted center shift vector of every point.
        l0_points = xyz[:, dataset_D:, :] # [B, C, N]
        l0_xyz = xyz[:, :dataset_D, :] # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #go through the first set abstraction layer, where the feature dimension goes from C to 64
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #go through the second set abstraction layer, where the feature dimension goes from 64 to 256

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #go through the first feature propagation layer, skip link l2 with l1
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) #go through the first feature propagation layer, skip link l1 with l0

        # Semantic segmentation branch, estimate the semantic information of every point.
        x = self.drop1(F.relu(self.bn1(self.conv11(l0_points))))
        x = self.conv12(x)
        x = F.log_softmax(x, dim=1)
        pred_sem_mat = x.permute(0, 2, 1)  # [B, N, nclass]

        # Center shift vector prediction branch, estimate the center shift vector of every point.
        # The "predicted center shift vector" represents "how big the distance between every point and corresponding geometry center of groundtruth instance
        # points group", such information could be ultilized later to "push" the point torwards the geometry center of groundtruth instance points group
        # before applying clustering algorithm, the points belong to same instance after such adjustment will be closer to each other thus easier to be
        # clustered. See more info of this idea in section 3.1. of paper: 2021.Hierarchical Aggregation for 3D Instance Segmentation.
        # Beware since the final purpose of training "PointNet++ for semantic segmentation network" is not only estimating the semantic information of
        # every radar detection point, but also applying the point with estimated semnatic information to clustering algorithm for better clustering. Thus
        # we employ this branch to estimate center shift vector of every point.
        x = self.drop2(F.relu(self.bn2(self.conv21(l0_points))))
        x = self.conv22(x)  # [B, C+D, N]
        pred_center_shift_vectors = x.permute(0, 2, 1)  # [B, N, C+D]

        return pred_sem_mat, pred_center_shift_vectors

class get_self_attention_pointnet2_for_semantic_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample, second_layer_npoint,
                 second_layer_radius, second_layer_nsample, group_conv=False):
        super(get_self_attention_pointnet2_for_semantic_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.self_attention_1 = self_attention(64)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.self_attention_2 = self_attention(256)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.self_attention_3 = self_attention(32)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])
            self.self_attention_4 = self_attention(16)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.self_attention_1 = self_attention(64)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.self_attention_2 = self_attention(256)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.self_attention_3 = self_attention(32)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])
            self.self_attention_4 = self_attention(16)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同

    def forward(self, xyz, dataset_D):
        # The PointNet++ based semantic segmentation network consists two branches: One for outputting predicted semantic information of every point,
        # one for outputting predicted center shift vector of every point.
        l0_points = xyz[:, dataset_D:, :] # [B, C, N]
        l0_xyz = xyz[:, :dataset_D, :] # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #go through the first set abstraction layer, where the feature dimension goes from C to 64
        l1_points = self.self_attention_1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #go through the second set abstraction layer, where the feature dimension goes from 64 to 256
        l2_points = self.self_attention_2(l2_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #go through the first feature propagation layer, skip link l2 with l1
        l1_points = self.self_attention_3(l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) #go through the first feature propagation layer, skip link l1 with l0
        l0_points = self.self_attention_4(l0_points)

        # Semantic segmentation branch, estimate the semantic information of every point.
        x = self.drop1(F.relu(self.bn1(self.conv11(l0_points))))
        x = self.conv12(x)
        x = F.log_softmax(x, dim=1)
        pred_sem_mat = x.permute(0, 2, 1)  # [B, N, nclass]

        # Center shift vector prediction branch, estimate the center shift vector of every point.
        # The "predicted center shift vector" represents "how big the distance between every point and corresponding geometry center of groundtruth instance
        # points group", such information could be ultilized later to "push" the point torwards the geometry center of groundtruth instance points group
        # before applying clustering algorithm, the points belong to same instance after such adjustment will be closer to each other thus easier to be
        # clustered. See more info of this idea in section 3.1. of paper: 2021.Hierarchical Aggregation for 3D Instance Segmentation.
        # Beware since the final purpose of training "PointNet++ for semantic segmentation network" is not only estimating the semantic information of
        # every radar detection point, but also applying the point with estimated semnatic information to clustering algorithm for better clustering. Thus
        # we employ this branch to estimate center shift vector of every point.
        x = self.drop2(F.relu(self.bn2(self.conv21(l0_points))))
        x = self.conv22(x)  # [B, C+D, N]
        pred_center_shift_vectors = x.permute(0, 2, 1)  # [B, N, C+D]

        return pred_sem_mat, pred_center_shift_vectors

class get_gmlp_based_pointnet2_for_semantic_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample,
                 second_layer_npoint, second_layer_radius, second_layer_nsample, number_of_points_in_a_frame, tiny_attn=False, group_conv=False):
        super(get_gmlp_based_pointnet2_for_semantic_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.gmlp1 = gmlp_block(64, 128, first_layer_npoint, tiny_attn, group_conv)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.gmlp2 = gmlp_block(256, 512, second_layer_npoint, tiny_attn, group_conv)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.gmlp3 = gmlp_block(32, 64, first_layer_npoint, tiny_attn, group_conv)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])
            self.gmlp4 = gmlp_block(16, 32, number_of_points_in_a_frame, tiny_attn, group_conv)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            # self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同
            self.conv22 = nn.Conv1d(16, 3, 1)
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.gmlp1 = gmlp_block(64, 32, first_layer_npoint, tiny_attn)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.gmlp2 = gmlp_block(256, 128, second_layer_npoint, tiny_attn)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.gmlp3 = gmlp_block(32, 16, first_layer_npoint, tiny_attn)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])
            self.gmlp4 = gmlp_block(16, 8, number_of_points_in_a_frame, tiny_attn)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, 3, 1)  # 输出的center shift vectors维度与原始detection points的维度相同

            # self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同

    def forward(self, xyz, dataset_D):
        # The PointNet++ based semantic segmentation network consists two branches: One for outputing predicted semantic information of every point, 
        # one for outputing predicted center shift vector of every point.
        # 在每层SA或FP后，添加一个gMLP block
        l0_points = xyz[:,dataset_D:,:] # [B, C, N]
        l0_xyz = xyz[:,:dataset_D,:] # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #go through the first set abstraction layer, where the feature dimension goes from C to 64
        l1_points = self.gmlp1(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #go through the second set abstraction layer, where the feature dimension goes from 64 to 256
        l2_points = self.gmlp2(l2_points.permute(0, 2, 1)).permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #go through the first feature propogation layer, skip link l2 with l1
        l1_points = self.gmlp3(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) #go through the first feature propogation layer, skip link l1 with l0
        l0_points = self.gmlp4(l0_points.permute(0, 2, 1)).permute(0, 2, 1)

        # Semantic segmentation branch, estimate the semantic information of every point.
        x = self.drop1(F.relu(self.bn1(self.conv11(l0_points))))
        x = self.conv12(x)
        x = F.log_softmax(x, dim=1)
        pred_sem_mat = x.permute(0, 2, 1)  # [B, N, nclass]

        # Center shift vector prediction branch, estimate the center shift vector of every point.
        # The "predicted center shift vector" represents "how big the distance between every point and corresponding geometry center of groundtruth instance 
        # points group", such information could be ultilized later to "push" the point torwards the geometry center of groundtruth instance points group 
        # before applying clustering algorithm, the points belong to same instance after such adjustment will be closer to each other thus easier to be 
        # clustered. See more info of this idea in section 3.1. of paper: 2021.Hierarchical Aggregation for 3D Instance Segmentation.
        # Beware since the final purpose of training "PointNet++ for semantic segmentation network" is not only estimating the semantic information of 
        # every radar detection point, but also applying the point with estimated semnatic information to clustering algorithm for better clustering. Thus
        # we employ this branch to estimate center shift vector of every point.
        x = self.drop2(F.relu(self.bn2(self.conv21(l0_points))))
        x = self.conv22(x)  # [B, C+D, N]
        pred_center_shift_vectors = x.permute(0, 2, 1)  # [B, N, C+D]

        return pred_sem_mat, pred_center_shift_vectors

class get_external_attention_based_pointnet2_for_semantic_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample, second_layer_npoint,
                 second_layer_radius, second_layer_nsample, group_conv=False):
        super(get_external_attention_based_pointnet2_for_semantic_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.ea1 = External_Attention(64, 64)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.ea2 = External_Attention(256, 64)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.ea3 = External_Attention(32, 64)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])
            self.ea4 = External_Attention(16, 64)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.ea1 = External_Attention(64, 64)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.ea2 = External_Attention(256, 64)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.ea3 = External_Attention(32, 64)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])
            self.ea4 = External_Attention(16, 64)

            # Semantic segmentation branch
            self.conv11 = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv12 = nn.Conv1d(16, num_class, 1)

            # Center shift vector prediction branch
            self.conv21 = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv22 = nn.Conv1d(16, dataset_D + dataset_C, 1)  # 输出的center shift vectors维度与原始detection points的维度相同

    def forward(self, xyz, dataset_D):
        # The PointNet++ based semantic segmentation network consists two branches: One for outputting predicted semantic information of every point,
        # one for outputting predicted center shift vector of every point.
        l0_points = xyz[:, dataset_D:, :] # [B, C, N]
        l0_xyz = xyz[:, :dataset_D, :] # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #go through the first set abstraction layer, where the feature dimension goes from C to 64
        l1_points = self.ea1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #go through the second set abstraction layer, where the feature dimension goes from 64 to 256
        l2_points = self.ea2(l2_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #go through the first feature propagation layer, skip link l2 with l1
        l1_points = self.ea3(l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) #go through the first feature propagation layer, skip link l1 with l0
        l0_points = self.ea4(l0_points)

        # Semantic segmentation branch, estimate the semantic information of every point.
        x = self.drop1(F.relu(self.bn1(self.conv11(l0_points))))
        x = self.conv12(x)
        x = F.log_softmax(x, dim=1)
        pred_sem_mat = x.permute(0, 2, 1)  # [B, N, nclass]

        # Center shift vector prediction branch, estimate the center shift vector of every point.
        # The "predicted center shift vector" represents "how big the distance between every point and corresponding geometry center of groundtruth instance
        # points group", such information could be ultilized later to "push" the point torwards the geometry center of groundtruth instance points group
        # before applying clustering algorithm, the points belong to same instance after such adjustment will be closer to each other thus easier to be
        # clustered. See more info of this idea in section 3.1. of paper: 2021.Hierarchical Aggregation for 3D Instance Segmentation.
        # Beware since the final purpose of training "PointNet++ for semantic segmentation network" is not only estimating the semantic information of
        # every radar detection point, but also applying the point with estimated semnatic information to clustering algorithm for better clustering. Thus
        # we employ this branch to estimate center shift vector of every point.
        x = self.drop2(F.relu(self.bn2(self.conv21(l0_points))))
        x = self.conv22(x)  # [B, C+D, N]
        pred_center_shift_vectors = x.permute(0, 2, 1)  # [B, N, C+D]

        return pred_sem_mat, pred_center_shift_vectors

class get_loss(nn.Module):
    # Calculate the loss for PointNet++ based semantic segmentation using radar detection points. However, beware 
    # the "center shift vector loss" is used for clustering of points with semantic information(using PointNet++ 
    # based semantic segmentation network to estimate the semantic information of every point, then apply clustering
    # on the points with estimated semnatic information.).
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, points, pred_sem_mat, pred_center_shift_vectors, label, device):
        instance_gt = label[:, 1, :]  # [B, N]
        class_gt = label[:, 0, :]

        B = pred_sem_mat.shape[0]
        # Loss part 1: 计算语义分割的loss
        sem_loss = 0
        for frame_id in range(B):
            sem_loss += F.nll_loss(pred_sem_mat[frame_id], class_gt[frame_id].long().to(device))
        sem_loss = sem_loss / B

        # Loss part 2: 计算center shift vector loss，详见“Hierarchical Aggregation for 3D Instance Segmentation”文章第3.1节公式（1）
        # 计算实际的center shift vectors
        center_shift_vectors_gt = torch.zeros_like(pred_center_shift_vectors).to(device)  # [B, N, C+D]
        for frame_id in range(B):
            num_instance_gt = int(max(instance_gt[frame_id]) + 1)
            for instance_id in range(num_instance_gt):  # 对于每一个实例
                points_loc_of_this_instance = torch.where(instance_gt[frame_id] == instance_id)[0]  # 所有属于该实例点的位序
                points_of_this_instance = points[frame_id, points_loc_of_this_instance]  # 所有属于该实例的点 [Npoints, C+D]
                center_of_this_instance = points_of_this_instance.mean(dim=0)  # 该实例中心点坐标 [C+D]
                center_shift_vectors_gt[frame_id, points_loc_of_this_instance] = center_of_this_instance - points_of_this_instance

        # 2.1. The 1st method to calculate center shift vector loss: 计算每个点的center shift vector loss by using L2 loss.
        # （如采用此计算方法，loss项权重为0.2）
        # 计算权重w
        weight = center_shift_vectors_gt.norm(dim=2).clamp(max=1)  # 与1比较取最小值 [B, N]
        center_shift_vectors_loss = (center_shift_vectors_gt - pred_center_shift_vectors).norm(dim=2) * weight  # [B, N]
        center_shift_vectors_loss = center_shift_vectors_loss.mean()

        # 2.2. The 2nd method to calculate center shift vector loss: using cosine similarity and "inner_product_between_predicted_and_gt_center_shift_vector/norm_of_gt_center_shift_vector^2"
        # （如采用此计算方法，最终的loss项权重为1）
        '''inner_product = center_shift_vectors_gt.matmul(pred_center_shift_vectors.permute((0, 2, 1))).diagonal(dim1=1, dim2=2)  # 对应center shift vectors作内积
        # [B,N,C]*[B,C,N]=[B,N,N]，由于实际上只需要center_shift_vectors_gt和pred_center_shift_vectors的同一行向量做内积，所以取对角元素得到[B,N]
        norm_gt = center_shift_vectors_gt.norm(dim=2)  # 实际center shift vectors的模长，[B,N]
        # norm_pred = pred_center_shift_vectors.norm(dim=2)  # [B,N]
        # cos_sim = inner_product / (norm_gt * norm_pred)
        cos_sim = F.cosine_similarity(pred_center_shift_vectors, center_shift_vectors_gt, dim=2)  # [B,N]
        inner_product_loss = (inner_product/(norm_gt * norm_gt + 1e-5) - 1).abs()
        # 由于期望的内积为groundtruth模长的平方，故考虑去其差值并归一化作为loss
        cos_sim_loss = 1 - cos_sim  # 期望的cos_sim值为1（两者同方向）
        center_shift_vectors_loss = inner_product_loss.mean() + cos_sim_loss.mean()  # 平衡数量级
        # print(inner_product_loss.mean().item(), cos_sim_loss.mean().item(), sem_loss.item())'''

        # 2.3. The 3rd method to calculate center shift vector loss: 使用cosine similarity和模长计算loss（如采用此计算方法，最终的loss项权重为1）
        '''cos_sim = F.cosine_similarity(pred_center_shift_vectors, center_shift_vectors_gt, dim=2)  # [B,N]
        cos_sim_loss = 1 - cos_sim  # 期望的cos_sim值为1（两者同方向）
        norm_gt = center_shift_vectors_gt.norm(dim=2)  # 实际center shift vectors的模长,[B,N]
        norm_pred = pred_center_shift_vectors.norm(dim=2)  # 预测center shift vectors的模长,[B,N]
        norm_loss = (norm_pred - norm_gt).abs()
        center_shift_vectors_loss = 0.2 * norm_loss.mean() + cos_sim_loss.mean()
        # print(norm_loss.mean().item(), cos_sim_loss.mean().item(), sem_loss.item())'''

        # 2.4. The 4th method to calculate center shift vector loss: 使用马氏距离计算loss（如采用此计算方法，loss项权重为0.2）
        '''center_shift_vectors_loss = mahalanobis_distance(pred_center_shift_vectors, center_shift_vectors_gt, device)  # [B, N]
        center_shift_vectors_loss = center_shift_vectors_loss.mean()'''

        total_loss = sem_loss + 0.2 * center_shift_vectors_loss  # 平衡两者数量级
        return total_loss

def mahalanobis_distance(pred_center_shift_vectors, center_shift_vectors_gt, device):
    '''
        计算pred_center_shift_vectors和center_shift_vectors_gt对应点的马氏距离
    '''
    B, N, _ = pred_center_shift_vectors.shape
    dist = torch.zeros((B, N)).to(device)
    with torch.no_grad():
        for frame_id in range(B):
            # 计算协方差矩阵
            pred_center_shift_vectors_of_this_instance = pred_center_shift_vectors[frame_id]
            center_shift_vectors_gt_of_this_instance = center_shift_vectors_gt[frame_id]
            Cov = cov(center_shift_vectors_gt_of_this_instance)  # 求协方差矩阵[C,C]
            if torch.matrix_rank(Cov) < Cov.shape[0]:  # 协方差矩阵不可逆
                # 对数据做PCA，去掉特征值0的维度
                eig_val, eig_vec = Cov.eig(eigenvectors=True)  # 计算特征值和特征向量
                eig_val = eig_val[:, 0]  # 由于协方差矩阵为半正定的（实对称），特征值一定为实数，故取特征值的实部
                index = (-eig_val).argsort()  # eig_val从大到小排列
                index = index[eig_val[index] > 1e-3]  # 需要特征值大于0的维度
                P = eig_vec[:, index]  # 降维矩阵P:[C,C']
                pred_center_shift_vectors_of_this_instance = pred_center_shift_vectors_of_this_instance.matmul(P)  # 降维 Y = XP; Y:[N,C']
                center_shift_vectors_gt_of_this_instance = center_shift_vectors_gt_of_this_instance.matmul(P)
                Cov = cov(pred_center_shift_vectors_of_this_instance)  # 重新计算协方差矩阵
            invCov = Cov.inverse()  # 协方差逆矩阵
            tmp = pred_center_shift_vectors_of_this_instance - center_shift_vectors_gt_of_this_instance  # [N,C]
            dist[frame_id] = tmp.matmul(invCov).matmul(tmp.permute(1, 0)).diagonal().sqrt()
            # [N,C']*[C',C']*[C',N]=[N,N]->N（由于只需要对应center shift vector计算马氏距离，因此取对角线元素）
    return dist

def cov(points):
    '''
        计算点集的协方差矩阵，points dim=[N,C]
    '''
    N = points.shape[0]
    mean = points.mean(dim=0)  # [C]
    points = points - mean
    Cov = points.permute(1, 0).matmul(points) / N  # [C,N]*[N,C]=[C,C]
    return Cov
