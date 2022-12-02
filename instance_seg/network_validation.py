import os
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.parallel
from scipy.stats import stats
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from torchvision import transforms, utils
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


'''
Semantic segmentation network using semantic segmentation metrics
'''
def validation_metric_for_semantic_segmentation(model, dataLoader, dataset_D, device):
    '''
        Perform predictions on the dataset and calculate the metrics.
    '''
    acc_total = 0
    f1score_total = 0
    mIoU_total = 0
    n_frames = 0
    for points, target in tqdm(dataLoader, desc='Calculating Accuracy, F1_score and mIoU on dataset'):
        with torch.no_grad():  # to turn off calculation of gradient when we aren't in training phase.
            points = points.permute(0, 2, 1)
            points, target = points.float().to(device), target[:, 0, :].float().to(device)
            semantic_segmentor = model.eval()  # Put network in evaluation mode (this is compulsory step when validation/validation,
            # to turn off something should NOT be used in validation, e.g. dropout, batch normalization )
            pred, _ = semantic_segmentor(points, dataset_D)  # Run the "trained semantic_segmentor model" and get the predicted
            # log_softmax value of each of classes for batch_size frames(i.e. pred), size of pred is batch_size x num_class(e.g. 600 x 4)
            acc_for_this_batch, f1score_for_this_batch, mIoU_for_this_batch = semantic_segmentation_metrics_accuracy(target, pred)
            # 由于最后一个batch可能不到batch_size帧，不应该使用各个batch的acc求平均，而应该使用每帧的acc求平均
            B = points.shape[0]
            acc_total += acc_for_this_batch * B  # 该batch中每帧acc之和
            f1score_total += f1score_for_this_batch * B
            mIoU_total += mIoU_for_this_batch * B
            n_frames += B
    acc = acc_total / n_frames
    f1score = f1score_total / n_frames
    mIoU = mIoU_total / n_frames
    return acc, f1score, mIoU

def mIoU_for_points_based_semantic_segmentation(target, pred_choice, nclass):
    '''
        计算每个class的IoU并求平均(label为数字id),target=[B,N]
    '''
    n_tp = torch.zeros(nclass)  # true positive
    n_fp = torch.zeros(nclass)  # false positive
    n_fn = torch.zeros(nclass)  # false negative
    for batch_idx in range(target.shape[0]):  # range(B)
        for idx in range(target.shape[1]):  # range(N)
            label_gt = int(target[batch_idx][idx])
            label_pred = int(pred_choice[batch_idx][idx])
            if label_gt == label_pred:
                n_tp[label_gt] += 1
            else:
                n_fn[label_gt] += 1  # 对于label_gt类，实际为真而预测为假，属于该类的FN
                n_fp[label_pred] += 1  # 对于label_pred类，实际为假而预测为真，属于该类的FP
    IoU = n_tp * 1. / (n_tp + n_fp + n_fn)  # 如果n_tp + n_fp + n_fn==0，结果为NaN
    return np.nanmean(IoU)

def semantic_segmentation_metrics_accuracy(target, pred):  # 具体计算Accuracy，pred=[B,N,nclass]
    nclass = pred.shape[2]
    pred_choice = pred.max(2)[1]
    # Get indices of the maximum log_softmax value(corresponds to probability), which will be used as predicted class(i.e. pred_choice)
    B = target.shape[0]
    target = target.data.cpu()
    pred_choice = pred_choice.data.cpu()
    acc = 0
    f1score = 0
    for batch_idx in range(B):
        acc += accuracy_score(target[batch_idx], pred_choice[batch_idx])  # 统计每帧的accuracy
        f1score += f1_score(target[batch_idx], pred_choice[batch_idx], average='weighted')  # 统计每帧的F1 score
    acc = acc / B  # 求整个batch的平均acc
    f1score = f1score / B  # 求整个batch的平均F1 score
    mIoU = mIoU_for_points_based_semantic_segmentation(target, pred_choice, nclass)  # 计算整个batch的mIoU
    return acc, f1score, mIoU


'''
Semantic segmentation network using instance segmentation metrics
'''
def validation_metric_for_semantic_segmentation_and_clustering(model, dataLoader, dataset_D, device):
    '''
        Perform predictions on the dataset and calculate the metrics.
    '''
    mmCov_total = 0
    mmAP_total = 0
    n_frames = 0
    for points, target in tqdm(dataLoader, desc='Calculating mmCov and mmAP on dataset'):
        with torch.no_grad():  # to turn off calculation of gradient when we aren't in training phase.
            points = points.permute(0, 2, 1)
            points, target = points.float().to(device), target.float().to(device)
            semantic_segmentor = model.eval()  # Put network in evaluation mode (this is compulsory step when validation/validation,
            # to turn off something should NOT be used in validation, e.g. dropout, batch normalization )
            pred_semmat, pred_center_shift_vectors = semantic_segmentor(points, dataset_D)
            points = points.permute(0, 2, 1)  # [B,N,C]
            points_shifted = points + pred_center_shift_vectors
            B, N, nclass = pred_semmat.shape
            for frame_id in range(B):
                shifted_points_of_this_frame = points_shifted[frame_id].view(1, N, -1)  # [1,N,C]
                label_of_this_frame = target[frame_id].view(1, 2, N)  # [1,2,N]
                pred_semmat_of_this_frame = pred_semmat[frame_id]  # [N,nclass]
                pred_class = pred_semmat_of_this_frame.max(1)[1]  # Get indices of the maximum log_softmax value, which will be used as predicted class
                conf_score = pred_semmat_of_this_frame.max(1)[0]  # Get the maximum log_softmax value, which will be used for mAP calculation
                pred_label = torch.stack((pred_class, conf_score), dim=0).reshape((1, 2, -1))  # [1,2,N]
                pred_instance = clustering_with_semantic_info(shifted_points_of_this_frame, pred_label, nclass)
                mCov_for_this_frame = mCov_for_clustering_with_semantic_information(label_of_this_frame, pred_label, pred_instance)
                mAP_for_this_frame = mAP_for_clustering_with_semantic_information(label_of_this_frame, pred_label, pred_instance, IoU_threashold=0.5)
                mmCov_total += mCov_for_this_frame
                mmAP_total += mAP_for_this_frame
            n_frames += B
    mmCov = mmCov_total / n_frames
    mmAP = mmAP_total / n_frames
    return mmCov, mmAP

def clustering_with_semantic_info(points, pred_label, nclass):  # 具体计算Accuracy，pred=[B,N,nclass]
    eps_list = [2.5, 1, 2, 2, 7]
    minpts_list = [1, 1, 1, 1, 1]
    pred_instance = {}  # keys: class_id; values: pred_class
    for class_id in range(nclass):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = points[0][mask]
        pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(features_class[:, :2].cpu())
        # Only using position info for DBSCAN.
        pred_instance[class_id] = pred_class
    return pred_instance

def mCov_for_clustering_with_semantic_information(label, pred_label, pred_instance):
    '''
        计算每个实例的平均覆盖率（mean coverage）,label=[1,2,N],pred_label=[1,1,N]
    '''
    class_gt = label[0, 0, :]
    instance_gt = label[0, 1, :]
    Cov_sum = 0
    N_instances = max(instance_gt) + 1
    for instance_id in range(int(N_instances)):  # 对于每一个groundtruth实例
        points_loc_of_this_instance = torch.where(instance_gt == instance_id)[0]  # 该实例所有点的位序
        num_points_of_this_instance = len(points_loc_of_this_instance)
        class_of_points_of_this_instance = class_gt[points_loc_of_this_instance[0]]  # 该实例所属类别
        pred_points_loc_of_this_class = torch.where(pred_label[0, 0, :] == int(class_of_points_of_this_instance))[0]  # 预测为该类的点的位序
        max_IoU = 0
        if int(class_of_points_of_this_instance) in pred_instance.keys():  # 预测没有点属于该类，IoU = 0
            pred_instances_of_this_class = pred_instance[int(class_of_points_of_this_instance)]  # 预测为该类的点的实例编号
            pred_num_instances_of_this_class = max(pred_instances_of_this_class) + 1
            for pred_instance_id in range(pred_num_instances_of_this_class):  # 对每一个预测实例
                points_loc_of_this_pred_instance = pred_points_loc_of_this_class[pred_instances_of_this_class == pred_instance_id]
                # 属于该预测实例的点的编号
                num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                num_union = len(set(list(points_loc_of_this_instance.cpu().numpy()) + list(points_loc_of_this_pred_instance.cpu().numpy())))
                num_intersection = num_points_of_this_instance + num_points_of_this_pred_instance - num_union
                IoU = num_intersection * 1. / num_union
                max_IoU = IoU if IoU > max_IoU else max_IoU
        Cov_sum += max_IoU
    mCov = Cov_sum / N_instances
    return float(mCov)

def mAP_for_clustering_with_semantic_information(label, pred_label, pred_instance, IoU_threashold=0.5):
    class_gt = label[0, 0, :]
    instance_gt = label[0, 1, :]
    pred_label = pred_label
    class_ids = set([int(class_id) for class_id in class_gt])
    AP_total = 0
    for class_id in class_ids:
        points_loc_of_this_class = torch.where(class_gt == class_id)[0]  # 该类所有点的位序
        instances_of_this_class = instance_gt[points_loc_of_this_class]  # 实际为该类的点的实例编号
        pred_points_loc_of_this_class = torch.where(pred_label[0, 0, :] == class_id)[0]  # 预测为该类的点的位序
        if class_id in pred_instance.keys():
            pred_instances_of_this_class = pred_instance[class_id]  # 预测为该类的点的实例编号
            pred_num_instances_of_this_class = max(pred_instances_of_this_class) + 1
            if pred_num_instances_of_this_class == 0:  # 该类点均被判断为噪声，AP=0（表现为AP_total不做改变）
                continue
            TP = np.zeros((2, pred_num_instances_of_this_class))  # 存放每个实例是否为TP及其属于该类的conf_score
            gt_instance_ids = set([int(instance_id) for instance_id in instances_of_this_class])
            for pred_instance_id in range(pred_num_instances_of_this_class):  # 对每一个预测实例
                points_loc_of_this_pred_instance = pred_points_loc_of_this_class[pred_instances_of_this_class == pred_instance_id]
                # 属于该预测实例的点的编号
                conf_score = pred_label[0, 1, points_loc_of_this_pred_instance].mean()
                TP[1, pred_instance_id] = conf_score
                num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                for instance_id in gt_instance_ids:  # 对于每一个groundtruth实例
                    points_loc_of_this_instance = torch.where(instance_gt == instance_id)[0]  # 该实例所有点的位序
                    num_points_of_this_instance = len(points_loc_of_this_instance)  # 该实例点的数量
                    num_union = len(set(list(points_loc_of_this_instance.cpu().numpy()) + list(points_loc_of_this_pred_instance.cpu().numpy())))
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


'''
Instance segmentation metric(for either instance segmentation network or clustering without point wise semantic information)
'''
def validation_metric_for_instance_segmentation(model, dataLoader, dataset_D, device, Th_s):
    '''
        Perform predictions on the dataset and calculate the metrics.
        Th_s: if S_ij < Th_s, then points pair P_i and P_j are in the same instance group
    '''
    mCov_total = 0
    mAP_total = 0
    n_frames = 0
    for points, label in tqdm(dataLoader, desc='Calculating mmCov and mmAP on dataset'):
        with torch.no_grad():  # to turn off calculation of gradient when we aren't in training phase.
            points = points.permute(0, 2, 1)  # [B, C, N]
            points, label = points.float().to(device), label.float().to(device)
            # Put network in evaluation mode (this is compulsory step when validation/validation, to turn off something
            # should NOT be used in validation, e.g. dropout, batch normalization)
            instance_segmentor = model.eval()
            # Implement instance segmentation.
            pred_simmat, pred_cfmat, pred_semmat = instance_segmentor(points, dataset_D)
            points = points.permute(0, 2, 1)  # [B, N, C]
            # Generate instance proposals by using predicted similarity matrix and "NMS" to merge some of instance proposals which have many overlap.
            pred_instance, pred_class = group_proposals_generation_and_merging(points, pred_simmat, pred_cfmat, pred_semmat, Th_s)
            mmCov_for_this_batch, mmAP_for_this_batch = instance_segmentation_metrics(label, pred_class, pred_instance)  # Calculate the metrics for validation data.
            # 由于最后一个batch可能不到batch_size帧，不应该使用各个batch的mmCov求平均，而应该使用每帧的mCov求平均
            B = points.shape[0]
            mCov_total += mmCov_for_this_batch * B  # 该batch中每帧mCov之和
            mAP_total += mmAP_for_this_batch * B
            n_frames += B
    mmCov = mCov_total / n_frames
    mmAP = mAP_total / n_frames
    return mmCov, mmAP

def group_proposals_generation_and_merging(points, sim_mat, cf_mat, sem_mat, Th_s, Th_c=0.4, gMLP=False):
    '''
        This is function is used to generate predicted proposals of group(instance) and impelemnt "per class merging of group(instance) 
        proposals" in the inference phase. In other word, it generates many possible predicted proposals of group(instance) by using 
        predicted similarity matrix, and implements per class based "NMS" for predicted group(instance) proposals where "the group(instace) 
        proposals under same predicted class label" which have enough overlapping points will be merged into one.
    '''
    sim_mat, cf_mat, sem_mat = sim_mat.cpu(), cf_mat.cpu(), sem_mat.cpu()
    B, N, _ = sim_mat.shape

    conf_valid_pts_mask = (cf_mat > Th_c)  # 满足confidence条件 [B,N]
    pred_conf_score, pred_class = sem_mat.max(dim=2)  # 每个点属于预测类别的conf score；每个点的预测类别 dim=[B,N]

    if gMLP == True:
        # 对于含有gMLP的方案，同一个点（被复制的点）的预测类别及其分数可能不同，需要处理
        for frame_id in range(B):
            mark = np.zeros(N)  # 标记是否处理过
            for idx in range(N):
                if mark[idx] == 1:  # 已处理的点不再处理
                    continue
                point = points[frame_id, idx, :]
                point_location = np.where(points[frame_id] == point)
                # 找到该点的所有位序（返回的是该元素每个坐标的位置，如第0个和第5个点都是，则返回([0,0,0,0,5,5,5,5],[0,1,2,3,0,1,2,3])）
                point_location = sorted(list(set(point_location[0])))  # 只取点的位序，把([0,0,0,0,5,5,5,5],[0,1,2,3,0,1,2,3])变成[0,5]
                mark[point_location] = 1  # 将这些点标记为已处理
                if len(point_location) != 1:  # 如果不止一个该元素
                    pred_class_of_these_points = pred_class[0, point_location]
                    pred_class_of_this_point = stats.mode(pred_class_of_these_points)[0][0]  # 取预测较多的一类作为该点类别
                    pred_class[0, point_location] = pred_class_of_this_point
                    pred_conf_score_of_these_points = sem_mat[0, point_location, pred_class_of_this_point].mean()  # 取所有conf score的平均值作为该点conf score
                    pred_conf_score[0, point_location] = pred_conf_score_of_these_points

    pred_instance = -1 * np.ones((B, N))  # 标记每个点属于哪一个instance
    group_class = []  # 标记每一个instance属于哪一类及其conf score，含有batch_size个元素，每个元素是[2,N_instance]的矩阵
    for frame_id in range(B):
        pred_instance_id = 0
        group_class_of_this_frame = []
        class_ids = np.unique(pred_class[frame_id])
        for class_id in class_ids:
            pts_in_this_class_mask = (pred_class[frame_id] == class_id)  # 属于该类的点  [N]
            valid_points_loc_of_this_class = np.where(pts_in_this_class_mask & conf_valid_pts_mask[frame_id])  # 属于该类的点且满足confidence条件的点的位序

            proposals = []  # 存放所有group proposal
            if valid_points_loc_of_this_class[0].shape[0] == 0:  # 当前类别下，没有有效的点
                proposals += [pts_in_this_class_mask]  # 把整个类别视为一个group_proposal
            else:
                for point_loc in valid_points_loc_of_this_class[0]:  # 对于该类别的每一个有效点
                    validpt_mask = (sim_mat[frame_id, point_loc] < Th_s[class_id]) & pts_in_this_class_mask  # 有效的group_proposal
                    if not validpt_mask.any():  # 如果该group proposal没有有效点，考察下一个
                        continue
                    same_as_before_flag = False  # 是否与之前的proposal代表同一个proposal
                    for proposal_id in range(len(proposals)):  # NMS
                        intersection_num = (validpt_mask & proposals[proposal_id]).sum()
                        union_num = (validpt_mask | proposals[proposal_id]).sum()
                        iou = float(intersection_num) / float(union_num)  # 当前proposal和之前得到的proposal计算iou
                        validpt_in_gp = float(intersection_num) / float(validpt_mask.sum())
                        # 当前proposal和之前得到的proposal交集的点数占当前proposal的点数比例
                        # TODO:有没有更好的方案，判断两个group proposal是否应该merge？
                        if iou > 0.6 or validpt_in_gp > 0.8:
                            same_as_before_flag = True
                            if validpt_mask.sum() > proposals[proposal_id].sum():  # 选择点数较多的proposal
                                proposals[proposal_id] = validpt_mask
                            continue
                    if same_as_before_flag is False:
                        proposals += [validpt_mask]  # 与之前的proposal不同，加入到proposal list中

            for proposal_id in range(len(proposals)):
                pred_instance[frame_id, proposals[proposal_id]] = pred_instance_id  # 标记第proposal_id个proposal中的点为第pred_instance_id个实例
                conf_score = pred_conf_score[frame_id, proposals[proposal_id]].mean()  # 各点conf_score的平均值
                group_class_of_this_frame.append([class_id, conf_score])  # 第pred_instance_id个实例的类别
                pred_instance_id += 1

        # 经过上一步骤，部分实例id可能因为被后来的proposal覆盖而不再存在，需要作出调整
        pred_instance_ids, cnt = np.unique(pred_instance[frame_id], return_counts=True)
        if -1 in pred_instance_ids:
            pred_instance_ids = list(pred_instance_ids)
            pred_instance_ids.remove(-1)
        pred_instance_this_frame = pred_instance[frame_id].copy()
        group_class_of_this_frame_new = []  # [N_instances, 2]
        for idx, instance_id in enumerate(pred_instance_ids):  # 例如pred_instance_ids = [0, 1, 3]
            pred_instance_this_frame[pred_instance[frame_id] == instance_id] = idx
            group_class_of_this_frame_new.append(group_class_of_this_frame[int(instance_id)])
        pred_instance[frame_id] = pred_instance_this_frame
        group_class_of_this_frame_new = np.array(group_class_of_this_frame_new).T  # [2, N_instances]
        group_class.append(group_class_of_this_frame_new)

        # 尽可能将所有点都分到某个group中（处理未被分到任何group的点）
        for idx, instance_id in enumerate(pred_instance[frame_id]):
            if instance_id == -1:  # 如果某点不属于任何group
                mask = (sim_mat[frame_id, idx] < Th_s[pred_class[frame_id, idx]])
                # 该点与所有点的similarity值小于该点预测类别的mask（表示该点可能与哪些点为同一group）
                points_instance_ids = pred_instance[frame_id, mask]  # 这些点的实例id
                valid_points_instance_ids = points_instance_ids[points_instance_ids != -1]  # 这些点中除掉不属于任何group的点
                if len(valid_points_instance_ids) != 0:  # 如果存在已经被分到某个group的，且可能与当前点为同一实例的点
                    pred_instance[frame_id, idx] = stats.mode(valid_points_instance_ids)[0][0]  # 将当前点分到多数点所属的实例中

    # Size of pred_instance is (B, N), size of group_class is (B, 2, number_of_predicted_instances) where 2 stands for the predicted class label and corresponding confidence for classification.
    return pred_instance, group_class

def get_similarity_distance_threshold(model, train_dataLoader, dataset_D, num_classes, device):
    '''
        The function which is used to calculate the similarity distance threshold, Th_s. This threshold will be used in the
        inference procedure to decide whether the element in i_th raw and j_th column of similarity matrix is small enough
        to judge the point i and point j belong to the same instance. See section 3.1. of paper "2018.SGPN: Similarity Group
        Proposal Network for 3D Point Cloud Instance Segmentation" for more detail.
        参考 https://github.com/laughtervv/SGPN/blob/master/valid.py 以及
        https://github.com/laughtervv/SGPN/blob/257f0458a5cd734db1642b8805d547dff72f9840/utils/test_utils.py#L11
    '''
    Ths = np.zeros(num_classes)  # 针对不同类别设置不同阈值
    cnt_class = np.zeros(num_classes)
    for points, label in tqdm(train_dataLoader, desc='Calculating similarity threshold'):
        with torch.no_grad():
            points = points.permute(0, 2, 1)  # [B, C, N]
            points, label = points.float().to(device), label.float().to(device)
            instance_segmentor = model.eval()
            pred_simmat, _, _ = instance_segmentor(points, dataset_D)  # [B,N,N]
            pred_simmat = pred_simmat.cpu()
            points, label = points.cpu(), label.cpu()
            class_gt = label[:, 0, :]  # [B, N]
            instance_gt = label[:, 1, :]  # [B, N]
            B, N, _ = points.shape
            for batch_id in range(B):
                for idx in range(N):
                    instance_id_of_this_point_in_groundtruth = instance_gt[batch_id, idx]  # 当前处理的点属于的实例编号
                    points_loc_of_this_instance = np.where(instance_gt[batch_id] == instance_id_of_this_point_in_groundtruth)[0]
                    # 和当前处理的点在当前batch中同属于一个实例的点在instance_gt[batch_id]中的index.
                    class_of_this_instance_in_groundtruth = int(class_gt[batch_id, points_loc_of_this_instance[0]])  # 当前处理的点属于的实例所属类别

                    points_of_this_instance_mask = (instance_gt[batch_id] == instance_id_of_this_point_in_groundtruth)
                    # 求出与当前处理的点属于同一个实例的所有点在instance_gt[batch_id]中的index.
                    points_of_other_instance_mask = (instance_gt[batch_id] != instance_id_of_this_point_in_groundtruth) \
                                                    & (class_gt[batch_id] == class_of_this_instance_in_groundtruth)
                    # 求出与当前处理的点属于同一类但属于不同实例的点的所有点在instance_gt[batch_id]中的index.

                    # 统计上述两类点的数量.
                    num_points_in_this_instance_in_groundtruth = np.array(points_of_this_instance_mask).sum()
                    num_points_out_this_instance_in_groundtruth = np.array(points_of_other_instance_mask).sum()

                    # 计算和当前处理的detection point(e.g. idx_th detection point)在当前batch中应该(在groundtruth中)同属于一个实例的所有
                    # detection points对应的similarity distance vector(i.e. the similarity distance between the current detection
                    # point and all other detection points which should belong to the same instance as the current detection point,
                    # according to groundtruth.)的直方图.
                    # Beware the output bins will be a array whose first element is the minimum value of the "similarity distances
                    # between current detection point and all other detection points which should belong to the same instance
                    # as the current detection point according to groundtruth"(in another word, the detection point which should
                    # be in the same instance as current processing detection point, has the most likelihood to be in the
                    # same instance as current processing detection point.) and the last element is the maximum value(Thus the
                    # detection point which should be in the same instance as current processing detection point, but has the
                    # least likelihood to be in the same instance as current processing detection point.).
                    # Also beware the size of predicted similarity matrix, pred_simmat, is (B, N, N).
                    hist, bins = np.histogram(pred_simmat[batch_id, idx, points_loc_of_this_instance], bins=20)

                    if num_points_out_this_instance_in_groundtruth > 0:
                        tp_over_fp = 0  # 最好的 tp/fp 值
                        id_bin_opt = -2
                        for id_bin, bin in enumerate(bins):
                            if bin == 0:    # 如果bin == 0，显然不可能以similarity distance == 0作为阈值，这样会导致没有true positive，所以要跳过。
                                break   # TODO: Should it be continue?
                            # 对于当前check的bin值，求出该bin值作为阈值时定义的TP(True positive)和FP(false positive)所占比例.
                            tp = float(np.array(pred_simmat[batch_id, idx, points_of_this_instance_mask] < bin).sum()) / float(num_points_in_this_instance_in_groundtruth)
                            fp = float(np.array(pred_simmat[batch_id, idx, points_of_other_instance_mask] < bin).sum()) / float(num_points_out_this_instance_in_groundtruth)
                            if tp <= 0.5:  # 当前的bin作为阈值来定义true positive和false positive将导致不足以得到足够的正例. 所以我们跳过该bin来check下个bin.
                                continue
                            if fp == 0. and tp > 0.5:
                                id_bin_opt = id_bin
                                break
                            if tp / fp > tp_over_fp:
                                tp_over_fp = tp / fp    # 对于每个bin值，如果得到的tp over fp比之前最大的tp_over_fp更大，则更新tp_over_fp.
                                id_bin_opt = id_bin     # 同时将该bin置为optimal bin.
                        if tp_over_fp > 4.:
                            # 如果最终遍历过所有的bin值, 得到的最大的tp_over_fp大于一个阈值(e.g. 4)，则选择对应的optimal bin作为class_of_this_instance_in_groundtruth
                            # 这个类别的instance的similarity distance的阈值.
                            # TODO: Think about whether it is proper to define per class similarity distance threshold, Th_s,
                            #  for our radar detection points case? If not, is there any better way to define the similarity
                            #  distance threshold?(For sure we should check the result first and see whether per class similarity
                            #  distance threshold works, then think about why it doesn't work in case it doesn't work)
                            Ths[class_of_this_instance_in_groundtruth] += bins[id_bin_opt]
                            cnt_class[class_of_this_instance_in_groundtruth] += 1
    for i in range(num_classes):  # 计算平均值
        if cnt_class[i] != 0:
            Ths[i] = Ths[i] / cnt_class[i]
        else:
            Ths[i] = 0.2
    return Ths

def mmCov_for_instance_segmentation(label, pred_classes, pred_instance):
    '''
        计算每个实例的平均覆盖率（mean coverage）, label=[B,2,N], pred_classes=[B,2,num_instances], pred_instance=[B,N]
    '''
    class_gt = label[:, 0, :]
    instance_gt = label[:, 1, :]
    B = label.shape[0]
    mmCov_total = 0
    for batch_id in range(B):
        Cov_sum = 0
        N_instances = max(instance_gt[batch_id]) + 1
        if len(pred_classes[batch_id]) != 0:  # len(pred_class)=0说明该帧没有预测实例，无需计算，该帧mCov为0
            pred_class = pred_classes[batch_id][0]
            for instance_id in range(int(N_instances)):  # 对于每一个groundtruth实例
                max_IoU = 0
                points_loc_of_this_instance = np.where(instance_gt[batch_id] == instance_id)[0]  # 该实例所有点的位序
                num_points_of_this_instance = len(points_loc_of_this_instance)  # 该实例点的数量
                class_of_this_instance_in_groundtruth = class_gt[batch_id, points_loc_of_this_instance[0]]  # 该实例所属类别
                pred_instance_of_this_class = np.where(pred_class == int(class_of_this_instance_in_groundtruth))[0]  # 属于此类的预测实例
                for pred_instance_id in pred_instance_of_this_class:  # 对属于同一类型的预测实例
                    points_loc_of_this_pred_instance = np.where(pred_instance[batch_id] == pred_instance_id)[0]  # 属于该预测实例的点的编号
                    num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                    num_union = len(set(list(points_loc_of_this_instance) + list(points_loc_of_this_pred_instance)))
                    num_intersection = num_points_of_this_instance + num_points_of_this_pred_instance - num_union
                    IoU = num_intersection * 1. / num_union
                    max_IoU = IoU if IoU > max_IoU else max_IoU
                Cov_sum += max_IoU
        mCov = Cov_sum / N_instances
        mmCov_total += mCov
    mmCov = mmCov_total / B
    return float(mmCov)

def mmAP_for_clustering_without_semantic_information(label, pred_classes, pred_instance, IoU_threashold=0.5):
    class_gt = label[:, 0, :]
    instance_gt = label[:, 1, :]

    B = label.shape[0]
    mAP_total = 0
    for batch_id in range(B):
        AP_total = 0
        class_ids = set([int(class_id) for class_id in class_gt[batch_id]])

        if len(pred_classes[batch_id]) != 0:  # len(pred_class)=0说明该帧没有预测实例，无需计算，该帧mAP为0
            N_pred_instances = int(max(pred_instance[batch_id]) + 1)  # 预测实例数
            pred_label = np.zeros_like(pred_instance[batch_id]) - 1  # 记录每个点对应的类别，初始化为全-1
            for instance_id in range(N_pred_instances):
                points_loc_of_this_pred_instance = np.where(pred_instance[batch_id] == instance_id)
                pred_label[points_loc_of_this_pred_instance] = pred_classes[batch_id][0, instance_id]  # 该实例的所有点标记为该实例所属类别

            for class_id in class_ids:
                points_loc_of_this_class = np.where(class_gt[batch_id] == class_id)[0]  # 该类所有点的位序
                instances_of_this_class = instance_gt[batch_id, points_loc_of_this_class]  # 实际为该类的点的实例编号
                pred_points_loc_of_this_class = np.where(pred_label == class_id)[0]  # 预测为该类的点的位序
                if class_id in pred_classes[batch_id][0]:
                    pred_instances_of_this_class = pred_instance[batch_id, pred_points_loc_of_this_class]  # 预测为该类的点的实例编号
                    pred_instance_ids = set([int(pred_instance_id) for pred_instance_id in pred_instances_of_this_class])
                    pred_num_instances_of_this_class = len(pred_instance_ids)
                    TP = np.zeros((2, pred_num_instances_of_this_class))  # 存放每个实例是否为TP及其属于该类的conf_score
                    gt_instance_ids = set([int(instance_id) for instance_id in instances_of_this_class])
                    for idx, pred_instance_id in enumerate(pred_instance_ids):  # 对每一个预测实例
                        points_loc_of_this_pred_instance = pred_points_loc_of_this_class[pred_instances_of_this_class == pred_instance_id]
                        # 属于该预测实例的点的编号
                        conf_score = pred_classes[batch_id][1, pred_instance_id]
                        TP[1, idx] = conf_score
                        num_points_of_this_pred_instance = len(points_loc_of_this_pred_instance)
                        for instance_id in gt_instance_ids:  # 对于每一个groundtruth实例
                            points_loc_of_this_instance = np.where(instance_gt[batch_id] == instance_id)[0]  # 该实例所有点的位序
                            num_points_of_this_instance = len(points_loc_of_this_instance)  # 该实例点的数量
                            num_union = len(set(list(points_loc_of_this_instance) + list(points_loc_of_this_pred_instance)))
                            num_intersection = num_points_of_this_instance + num_points_of_this_pred_instance - num_union
                            IoU = num_intersection * 1. / num_union
                            if IoU > IoU_threashold:
                                TP[0, idx] = 1  # 标记该估计实例为TP
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
        mAP_total += mAP
    mmAP = mAP_total / B
    return mmAP

def instance_segmentation_metrics(label, pred_class, pred_instance):
    label = label.cpu()
    mmCov = mmCov_for_instance_segmentation(label, pred_class, pred_instance)
    mmAP = mmAP_for_clustering_without_semantic_information(label, pred_class, pred_instance)
    return mmCov, mmAP

def metrics_confusion_matrix(target, pred):
    # Get indices of the maximum log_softmax value(corresponds to probability), which will be used as predicted class(i.e. pred_choice)
    pred_choice = pred.max(1)[1]
    target, pred_choice = target.cpu().numpy(), pred_choice.cpu().numpy()
    # Get the confusion matrix
    return confusion_matrix(target, pred_choice, labels=[0, 1,2,3,4,5])


'''
Set of methods for visualizing the results
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix in probability format'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix in probability")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax
