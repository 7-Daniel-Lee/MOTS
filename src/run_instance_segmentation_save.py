import joblib
import pickle
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from instance_seg.train_pointnets_for_semantic_segmentation_radar_scenes import parse_args
from instance_seg.pointnet2_sem_seg import get_pointnet2_for_semantic_segmentation_model, get_gmlp_based_pointnet2_for_semantic_segmentation_model,\
    get_external_attention_based_pointnet2_for_semantic_segmentation_model
from src.radar_scenes_val_test_generator import Radar_Scenes_Test_Dataset


'''
%% ----------------------------------- Run different approaches to implement instance segmentation for radar detection points ------------------------------ %%
This is script which runs different approaches to implement instance segmentation for radar detection points which provided in radarscenes dataset. 

%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] ST-DBSCAN: An algorithm for clustering spatial–temporal data
  [2] Modification of DBSCAN and application to rangeDopplerDoA measurements for pedestrian recognition with an automotive radar system
  [3] 2018.SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation
'''


"""
All the ultility functions as following
"""
def illustration_points(detection_points):
    x = detection_points[0][:, 0]
    y = detection_points[0][:, 1]
    plt.scatter(x, y, marker='o')
    plt.title('Original detection points')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.show()


def illustration_points_with_semantic_segmentation_and_clustering(detection_points, label, pred_label, pred_instance, selected_algorithm):
    marker_list = ['o', 'D', '^', '*', 's']
    class_list = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE']
    assert len(marker_list) == len(class_list)
    plt.figure(figsize=(12, 6))
    # illustrate groundtruth labels and instances
    plt.subplot(121)
    for class_id in range(len(class_list)):
        mask = label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]
        x = features_class[:, 0]
        y = features_class[:, 1]
        plt.scatter(x, y, c=label[0, 1, :][mask], marker=marker_list[class_id], label=class_list[class_id])
    plt.title('Groundtruth labels and instances\n(Different shapes represent different classes; different\ncolors in the same class represent different instances)')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    # For some reason, we have to comment out following line when running the code in vscode IDE, otherwise strange error
    # "Could not load source '<__array_function__ internals>': Source unavailable." happens. If we don't want to comment
    # out following line to avoid this problem, we could run this code in command line(e.g. cmd).
    plt.legend()

    # illustrate predicted labels and instances
    plt.subplot(122)
    for class_id in range(len(class_list)):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]
        x = features_class[:, 0]
        y = features_class[:, 1]
        mask = pred_instance[class_id] != -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c=pred_instance[class_id][mask], marker=marker_list[class_id], label=class_list[class_id])
        mask = pred_instance[class_id] == -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c='black', marker='x', label='NOISE')
    plt.title('Predicted instances by using\n' + selected_algorithm + '\n(Different colors represent different instances)')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    # For some reason, we have to comment out following line when running the code in vscode IDE, otherwise strange error
    # "Could not load source '<__array_function__ internals>': Source unavailable." happens. If we don't want to comment
    # out following line to avoid this problem, we could run this code in command line(e.g. cmd).
    plt.legend()
    plt.show()


def mCov_for_clustering_with_semantic_information(label, pred_label, pred_instance):
    '''
        计算每个实例的平均覆盖率（mean coverage）,label=[1,2,N],pred_label=[1,1,N]
    '''
    class_gt = label[0, 0, :]
    instance_gt = label[0, 1, :]
    Cov_sum = 0
    N_instances = max(instance_gt) + 1
    for instance_id in range(int(N_instances)):  # 对于每一个groundtruth实例
        points_loc_of_this_instance = np.where(instance_gt == instance_id)[0]  # 该实例所有点的位序
        num_points_of_this_instance = len(points_loc_of_this_instance)
        class_of_points_of_this_instance = class_gt[points_loc_of_this_instance[0]]  # 该实例所属类别
        pred_points_loc_of_this_class = np.where(pred_label[0, 0, :] == int(class_of_points_of_this_instance))[0]  # 预测为该类的点的位序
        max_IoU = 0
        if int(class_of_points_of_this_instance) in pred_instance.keys():  # 预测没有点属于该类，IoU = 0
            pred_instances_of_this_class = pred_instance[int(class_of_points_of_this_instance)]  # 预测为该类的点的实例编号
            pred_num_instances_of_this_class = max(pred_instances_of_this_class) + 1
            for pred_instance_id in range(pred_num_instances_of_this_class):  # 对每一个预测实例
                points_loc_of_this_pred_instance = pred_points_loc_of_this_class[pred_instances_of_this_class == pred_instance_id]  # 属于该预测实例的点的编号
                num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                num_union = len(set(list(points_loc_of_this_instance) + list(points_loc_of_this_pred_instance)))
                num_intersection = num_points_of_this_instance + num_points_of_this_pred_instance - num_union
                IoU = num_intersection * 1. / num_union
                max_IoU = IoU if IoU > max_IoU else max_IoU
        Cov_sum += max_IoU
    mCov = Cov_sum / N_instances
    return float(mCov)


def mAP_for_clustering_with_semantic_information(label, pred_label, pred_instance, IoU_threashold=0.5):
    class_gt = label[0, 0, :]
    instance_gt = label[0, 1, :]
    class_ids = set([int(class_id) for class_id in class_gt])
    AP_total = 0
    for class_id in class_ids:
        points_loc_of_this_class = np.where(class_gt == class_id)[0]  # 该类所有点的位序
        instances_of_this_class = instance_gt[points_loc_of_this_class]  # 实际为该类的点的实例编号
        pred_points_loc_of_this_class = np.where(pred_label[0, 0, :] == class_id)[0]  # 预测为该类的点的位序
        if class_id in pred_instance.keys():
            pred_instances_of_this_class = pred_instance[class_id]  # 预测为该类的点的实例编号
            pred_num_instances_of_this_class = max(pred_instances_of_this_class) + 1
            if pred_num_instances_of_this_class == 0:  # 该类点均被判断为噪声，AP=0（表现为AP_total不做改变）
                continue
            TP = np.zeros((2, pred_num_instances_of_this_class))  # 存放每个实例是否为TP及其属于该类的conf_score
            gt_instance_ids = set([int(instance_id) for instance_id in instances_of_this_class])
            for pred_instance_id in range(pred_num_instances_of_this_class):  # 对每一个预测实例
                points_loc_of_this_pred_instance = pred_points_loc_of_this_class[pred_instances_of_this_class == pred_instance_id]  # 属于该预测实例的点的编号
                conf_score = np.mean(pred_label[0, 1, points_loc_of_this_pred_instance])
                TP[1, pred_instance_id] = conf_score
                num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                for instance_id in gt_instance_ids:  # 对于每一个groundtruth实例
                    points_loc_of_this_instance = np.where(instance_gt == instance_id)[0]  # 该实例所有点的位序
                    num_points_of_this_instance = len(points_loc_of_this_instance)  # 该实例点的数量
                    num_union = len(set(list(points_loc_of_this_instance) + list(points_loc_of_this_pred_instance)))
                    num_intersection = num_points_of_this_instance + num_points_of_this_pred_instance - num_union
                    IoU = num_intersection * 1. / num_union
                    if IoU > IoU_threashold:
                        TP[0, pred_instance_id] = 1  # 标记该估计实例为TP
                        break
            TP = TP[:, np.argsort(-TP[1])]  # 按照第2行（conf_score)由大到小排序
            PR = np.zeros((pred_num_instances_of_this_class, 2))
            current_TP = 0
            for idx in range(pred_num_instances_of_this_class):  # 依次计算Precision和Recall值
                n = idx + 1  # 已经遍历的数量
                if TP[0, idx] == 1:
                    current_TP += 1
                PR[idx, 0] = current_TP / n  # precision查准率
                PR[idx, 1] = current_TP / len(gt_instance_ids)  # recall查全率
            for idx in range(pred_num_instances_of_this_class):  # 平滑操作
                PR[idx, 0] = max(PR[idx:, 0])
            # print(PR.shape)
            PR = np.row_stack(([PR[0, 0], 0], PR))  # 添加一行，防止该类只有1个预测实例导致下面无法计算；同时防止因为第一个recall不为0导致漏算面积
            AP = 0
            for idx in range(pred_num_instances_of_this_class):  # 计算AP值（面积）
                AP += (PR[idx, 0] + PR[idx + 1, 0]) * (PR[idx + 1, 1] - PR[idx, 1]) / 2.
        else:  # 预测没有点属于该类，AP = 0
            AP = 0
        AP_total += AP
    mAP = AP_total / len(class_ids)
    return mAP


# # Apply trained PointNet++ for points semantic segmentation for all points.
def pretrained_pointnet2_for_semantic_segmentation_model(model, dataLoader, dataset_D, device):
    for duplicated_detection_points_with_uuid, label in dataLoader:
        # print(duplicated_detection_points[0, :20, :])
        if duplicated_detection_points_with_uuid.numel() == 1:
            yield None
        else:
            with torch.no_grad():
                duplicated_detection_points = duplicated_detection_points_with_uuid[:, :, 0:4]    # 断点看下uuid在哪!!
                # print(duplicated_detection_points)
                duplicated_detection_points = duplicated_detection_points.permute(0, 2, 1)  # [B, C, N]
                duplicated_detection_points = duplicated_detection_points.float().to(device)
                detection_points_semantic_segmentor = model.eval()  # Put network in evaluation mode
                pred_label, pred_center_shift_vectors = detection_points_semantic_segmentor(duplicated_detection_points, dataset_D)
                # Run the "trained detection_points_semantic_segmentor model" and get the predicted log_softmax and center shift vectors
                # value of each of classes for batch_size frames. pred:[batch_size, num_class]
                pred_class = pred_label.max(2)[1]  # Get indices of the maximum log_softmax value, which will be used as predicted class
                conf_score = pred_label.max(2)[0]  # Get the maximum log_softmax value, which will be used for mAP calculation
                duplicated_detection_points = duplicated_detection_points.permute(0, 2, 1)  # [B, N, C]
                # print(duplicated_detection_points_with_uuid.shape, label.shape)  # label:[B, 2, N] pred:[1, N]
                pred_label = np.row_stack((pred_class, conf_score)).reshape((1, 2, -1))  # 与label保持结构一致
                duplicated_detection_points_with_semantic_information = (duplicated_detection_points_with_uuid, label, pred_label, pred_center_shift_vectors)
                yield duplicated_detection_points_with_semantic_information


# def pretrained_pointnet2_for_semantic_segmentation_model(model, dataLoader, dataset_D, device):
#     for duplicated_detection_points_with_uuid, label in dataLoader:
#         # print(duplicated_detection_points[0, :20, :])
#         if duplicated_detection_points_with_uuid.numel() == 1:
#             yield None
#         else:
#             with torch.no_grad():
#                 duplicated_detection_points = duplicated_detection_points_with_uuid[:, :, 0:4]    # 断点看下uuid在哪!!
#                 # print(duplicated_detection_points)
#                 duplicated_detection_points = duplicated_detection_points.permute(0, 2, 1)  # [B, C, N]
#                 duplicated_detection_points = duplicated_detection_points.float().to(device)
#                 detection_points_semantic_segmentor = model.eval()  # Put network in evaluation mode
#                 pred_label, pred_center_shift_vectors = detection_points_semantic_segmentor(duplicated_detection_points, dataset_D)
#                 # Run the "trained detection_points_semantic_segmentor model" and get the predicted log_softmax and center shift vectors
#                 # value of each of classes for batch_size frames. pred:[batch_size, num_class]
#                 pred_class = pred_label.max(2)[1]  # Get indices of the maximum log_softmax value, which will be used as predicted class
#                 conf_score = pred_label.max(2)[0]  # Get the maximum log_softmax value, which will be used for mAP calculation
#                 duplicated_detection_points = duplicated_detection_points.permute(0, 2, 1)  # [B, N, C]
#                 # print(duplicated_detection_points_with_uuid.shape, label.shape)  # label:[B, 2, N] pred:[1, N]
#                 pred_label = np.row_stack((pred_class, conf_score)).reshape((1, 2, -1))  # 与label保持结构一致
#                 duplicated_detection_points_with_semantic_information = (duplicated_detection_points, label, pred_label, pred_center_shift_vectors)
#                 yield duplicated_detection_points_with_semantic_information



# Remove all the duplicate detection points with semantic information
def remove_duplication_detection_points_with_semantic_information(duplicated_detection_points_with_semantic_information):
    duplicated_detection_points, label, pred_label, pred_center_shift_vectors = duplicated_detection_points_with_semantic_information
    C = duplicated_detection_points.shape[2]  # [B, N, C]
    idx = 0
    while idx < duplicated_detection_points.shape[1]:
        point = duplicated_detection_points[0, idx, :]
        point_location = np.where(duplicated_detection_points[0] == point)
        # 找到该点的位序（返回的是该元素每个坐标的位置，如第0个和第5个点都是，则返回([0,0,0,0,5,5,5,5],[0,1,2,3,0,1,2,3])）
        point_location = sorted(list(set(point_location[0])))  # 只取点的位序，把([0,0,0,0,5,5,5,5],[0,1,2,3,0,1,2,3])变成[0,5]
        if len(point_location) != 1:  # 如果不止一个该元素
            duplicated_detection_points = np.delete(duplicated_detection_points[0], point_location[1:], axis=0).view(1, -1, C)
            # 去除所有相同点，仅保留当前点
            label = np.delete(label[0], point_location[1:], axis=1).view(1, 2, -1)
            pred_label = np.delete(pred_label[0], point_location[1:], axis=1).reshape((1, 2, -1))
            pred_center_shift_vectors = np.delete(pred_center_shift_vectors[0], point_location[1:], axis=0).view(1, -1, C-2)
        idx += 1
    detection_points_with_semantic_information = (duplicated_detection_points, label, pred_label, pred_center_shift_vectors)
    return detection_points_with_semantic_information


if __name__ == '__main__':
    """
    2nd approach for radar detection points instance segmentation: Apply pretrained PointNet++ for points semantic segmentation first, then run different DBSCAN 
    based clustering algorithms on radar detection points.
    """
    args = parse_args()
    radar_scenes_test_dataset_duplicated_detection_points = Radar_Scenes_Test_Dataset(args.datapath, transforms=None, sample_size=200, non_static=True)
    duplicated_detection_points_dataloader = DataLoader(radar_scenes_test_dataset_duplicated_detection_points, batch_size=1,
                                                        shuffle=False, num_workers=0)

    saveloadpath = 'trained_models/sem_seg.pth'

    # 使用和训练时相同的网络参数
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    if args.model_configuration == 'Pointnet2_for_Semantic_Segmentation':
        detection_points_semantic_segmentor = get_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D,
                                      args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                      args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                      args.turn_on_light_weighted_network_model_using_group_conv)
    elif args.model_configuration == 'Self_Attention_based_Pointnet2_for_Semantic_Segmentation':
        # detection_points_semantic_segmentor = get_self_attention_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D,
        #                               args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
        #                               args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
        #                               args.turn_on_light_weighted_network_model_using_group_conv)
        pass
    elif args.model_configuration == 'gMLP_based_Pointnet2_for_Semantic_Segmentation':
        detection_points_semantic_segmentor = get_gmlp_based_pointnet2_for_semantic_segmentation_model(args.numclasses,
                                      args.dataset_D, args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                      args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, 200,
                                      args.turn_on_light_weighted_network_model_using_group_conv)
    elif args.model_configuration == 'External_Attention_based_Pointnet2_for_Semantic_Segmentation':
        detection_points_semantic_segmentor = get_external_attention_based_pointnet2_for_semantic_segmentation_model(args.numclasses,
                                      args.dataset_D, args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                      args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                      args.turn_on_light_weighted_network_model_using_group_conv)
    detection_points_semantic_segmentor = detection_points_semantic_segmentor.to(device)
    checkpoint = torch.load(saveloadpath, map_location=device)
    detection_points_semantic_segmentor.load_state_dict(checkpoint['best_model_state_dict'])

    # Run DBSCAN at every frame with only using position information of every detection point, (x, y), as feature.
    selected_algorithm = 'PointNet++ based Semantic Segmentation + DBSCAN'
    mmCov = 0
    mmAP = 0
    frame_id = 0
    data_dict = {}
    duplicated_detection_points_with_semantic_information_for_all_frames = pretrained_pointnet2_for_semantic_segmentation_model(
                                detection_points_semantic_segmentor, duplicated_detection_points_dataloader, args.dataset_D, device)
    # iterate over frames
    frames = {}
    for duplicated_detection_points_with_semantic_information in tqdm(duplicated_detection_points_with_semantic_information_for_all_frames,
                                                                      total=len(duplicated_detection_points_dataloader)):
        if duplicated_detection_points_with_semantic_information is None:
            instances = []
        else:
            detection_points_with_semantic_information = remove_duplication_detection_points_with_semantic_information(duplicated_detection_points_with_semantic_information)
            detection_points, label, pred_label, pred_center_shift_vectors = detection_points_with_semantic_information
            # illustration_points(detection_points)
            eps_list = [2.5, 1, 2, 2, 7]
            minpts_list = [1, 1, 1, 1, 2]
            instances = {}  # keys: instance_id; values: dictionary of all the points and class_id
            start_ins_id = 0
            for class_id in range(args.numclasses):
                mask = pred_label[0, 0, :] == class_id
                if not mask.any():
                    continue
                features_class = detection_points[0][mask]  # 属于该类别的点
                features_shift_class = pred_center_shift_vectors[0][mask]  # 属于该类别的点的center shift vectors
                if args.estimate_center_shift_vectors == True:
                    # Use estimated center shift vector of each point to "push" the point torwards the geometry center of groundtruth instance points group, if args.estimate_center_shift_vectors == True. 
                    # The points belong to same instance after such adjustment will be closer to each other thus easier to be clustered.
                    # See more info of this idea in section 3.1. of paper: 2021.Hierarchical Aggregation for 3D Instance Segmentation.
                    pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(
                                                                                    features_class[:, :2] + features_shift_class[:, :2])
                else:
                    pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(features_class[:, :2])
                # Only using position info for DBSCAN.
                pred_class += 1
                # iterate over instance ID of the class
                for ins_id_class in range(1, max(pred_class)+1):
                    # generate instance ID of the frame
                    ins_id = ins_id_class + start_ins_id - 1
                    # extract all the points of that instance ID
                    idx = np.where(pred_class == ins_id_class)  
                    instances[ins_id] = {'class_ID': class_id, 'points':features_class[idx, :]}
                start_ins_id = max(pred_class)
        frames[frame_id] = instances
        frame_id += 1 
    # write to file
    np.save('data/SegmentSeq109_uuid.npy', frames)
