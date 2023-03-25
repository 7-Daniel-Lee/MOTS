"""
    SORT adpted version for tracking by instance segmentation
"""
from __future__ import print_function
from cmath import nan

import os
from copy import deepcopy
from sys import displayhook
from turtle import distance
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from logging import basicConfig, DEBUG, INFO
from tqdm import tqdm
import time
import argparse
from filterpy.kalman import KalmanFilter
from sklearn import cluster
from color_scheme import COLOR
from scipy.optimize import linear_sum_assignment
from distances import convex_hull_iou
from sort_instance_Euclidean import find_instance_from_prediction, get_cluster_centeroid, get_mean_doppler_velocity, KalmanBoxTracker, Sort, parse_args, get_cost


np.random.seed(0)

NUM_COLOR = len(COLOR)


def linear_assignment(cost_matrix):
  '''
  Hungarian algorithm to solve the MOT association problem
  '''
  x, y = linear_sum_assignment(cost_matrix)
  return np.array(list(zip(x, y)))


def iou_batch(segment_instances:List, tracked_instances:List):
  """
  From SORT: Computes IOU between two instances 
  segment_instances:
  tracked_instances: 
  return IOU matrix, the negative of cost matrix
  """
  num_seg = len(segment_instances) 
  num_track = len(tracked_instances)
  iou_matrix = np.zeros((num_seg, num_track))  #初始化iou_matrix 
  for row in range(num_seg):
    for col in range(num_track):
      iou_matrix[row, col] = convex_hull_iou(segment_instances[row].numpy(), tracked_instances[col].numpy())   #get the minimum convex hull of two clusters, calculate their IoU
  return iou_matrix


def associate_detections_to_trackers(instances:List, trackers:List, iou_threshold = 0.3, distance_threshold=0.1):
  """
  Assigns instance segmentations to tracked object (both represented as clusters)
  param trackers: trackers' predictions in this frame

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(instances)), np.empty((0,5),dtype=int)  # change!  # 为什么中间那个是arange，其他两个是empty？
    # matched, unmatched_dets, unmatched_trks

  # here judge the shape of instances and trackers first, 
  # group them based on if there are more than 3 points
  instances = deepcopy(instances)
  trackers = deepcopy(trackers)
  '''
  深度复制，表示后期更改后不会影响现有复制的内容
  '''
  instances_less_2 = []  #小于两个点的instances 
  trackers_less_2 = []   #小于两个点的trackers 
  instances_idx = []
  trackers_idx = []
  for idx, instance in enumerate(instances):  #循环遍历每一帧的instances 
    if instance.shape[1] < 3:
      instances_idx.append(idx)
      instances_less_2.append(instance)
  for idx in reversed(instances_idx):  #倒序遍历instances_idx 
    instances.pop(idx)  #弹出对应的帧数 
  for idx, tracker in enumerate(trackers):  #循环遍历每一帧的trackers 
    if tracker.shape[1] < 3:
      trackers_idx.append(idx)
      trackers_less_2.append(tracker)
  for idx in reversed(trackers_idx):  #倒序遍历trackers_idx 
    trackers.pop(idx)   #弹出对应的帧数 
  
  # associate intances that are less than three points
  euclidean_matrix = get_cost(instances_less_2, trackers_less_2)  #定义euclidean_matrix
  if min(euclidean_matrix.shape) > 0:
    a = (euclidean_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      eu_matched_indices = np.stack(np.where(a), axis=1)
    else:
      eu_matched_indices = linear_assignment(euclidean_matrix)   #通常情况下利用匈牙利算法data association 
  else:
    eu_matched_indices = np.empty(shape=(0,2))

  # 记录未匹配的检测框及跟踪框 
  # 未匹配的检测框放入eu_unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标  
  eu_unmatched_detections = []
  for d, det in enumerate(instances_less_2):  #循环遍历每一帧中的instances_less_2 
    if(d not in eu_matched_indices[:,0]):#如果检测器中第d个检测结果不在匹配结果索引中，则d未匹配上 
      eu_unmatched_detections.append(d) # list of instances id
  eu_unmatched_trackers = []
  for t, trk in enumerate(trackers_less_2):  #循环遍历每一帧中的trackers_less_2 
    if(t not in eu_matched_indices[:,1]):#如果跟踪器中第t个跟踪结果不在匹配结果索引中，则t未匹配上 
      eu_unmatched_trackers.append(t)
  #filter out matched with large distance
  eu_matches = []  #存放过滤后的匹配结果 
  for m in eu_matched_indices:  #遍历粗匹配结果 
    if(euclidean_matrix[m[0], m[1]]>distance_threshold):  #m[0]是检测器ID， m[1]是跟踪器ID，如它们的欧几里得代价矩阵大于阈值则将它们视为未匹配成功 
      eu_unmatched_detections.append(m[0])
      eu_unmatched_trackers.append(m[1])
    else:
      eu_matches.append(m.reshape(1,2))  #将过滤后的匹配对维度变形成1x2形式 

  # associate instances that are more than three points
  #同理，利用iou_matrix进行数据关联 
  iou_matrix = iou_batch(instances, trackers)
  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        iou_matched_indices = np.stack(np.where(a), axis=1)
    else:
      iou_matched_indices = linear_assignment(-iou_matrix)
  else:
    iou_matched_indices = np.empty(shape=(0,2))

  iou_unmatched_detections = []
  for d, det in enumerate(instances):
    if(d not in iou_matched_indices[:,0]):
      iou_unmatched_detections.append(d) # list of instances id
  iou_unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in iou_matched_indices[:,1]):
      iou_unmatched_trackers.append(t)
  #filter out matched with low IOU
  iou_matches = []
  for m in iou_matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      iou_unmatched_detections.append(m[0])
      iou_unmatched_trackers.append(m[1])
    else:
      iou_matches.append(m.reshape(1,2))

  # concatenate最终的匹配结果和未匹配结果 
  matches = eu_matches + iou_matches
  unmatched_detections = eu_unmatched_detections + iou_unmatched_detections
  unmatched_trackers = eu_unmatched_trackers + iou_unmatched_trackers
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, distance_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.distance_threshold = distance_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, frame:dict):
    """
    Params:
    frame: dictionary 
      clusters - a list of ndarray 
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers, associate the detections with predictions  
    trks = []  
    to_del = []
    ret = []
    for t in range(len(self.trackers)):
      pred = self.trackers[t].predict(frame)['points'] # ndarray
      trks.append(pred)
      if np.any(np.isnan(pred.shape)):   # if any of the predictions is Nan, delete the tracker 
        to_del.append(t)
    for t in reversed(to_del):
      self.trackers.pop(t)

    if frame == []:
      clusters = [np.array([[[1e4, 1e4, 1e4, 1e4]]])]
      print('empty frame')
    else:
      clusters = [instance['points'] for instance in frame['seg instances'].values()] # a list of ndarray

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(clusters, trks, self.distance_threshold)
    # 可以分别做2次association,小于两个点的用距离，大于的用IOU 

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(clusters[m[0]].numpy())   # m is the index for matched clusters

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(clusters[i])   # one new tracker for each unmatched cluster   
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state(frame) 
        # rule-based track management 持续更新+ 连续match数量大于最小阈值或者还没到更新次数还没达到该阈值,最初几帧 
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # there are three trackers, but only one added in the output 
          ret.append((d, trk.id+1)) # 
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return ret
    return np.empty((0,5))


if __name__ == '__main__':
  # all train
  args = parse_args()
  args.seq_path = 'C:\\Users\\12283\\Desktop\\SCI\\Radar_PointNet_Panoptic_Tracking_and_Segmentation-main\\Radar_PointNet_Panoptic_Tracking_and_Segmentation-main\\src\\data_short'
  # display = args.display
  display = True
  phase = args.phase
  total_time = 0.0

  if not os.path.exists('output'):
    os.makedirs('output')
  # load segments
  segments_path = os.path.join(args.seq_path, phase, 'SegmentSeq109_trackid.npy')
  sequence_segments = np.load(segments_path, allow_pickle='TRUE')
  
  mot_tracker = Sort(max_age=args.max_age, 
                      min_hits=args.min_hits,
                      distance_threshold=args.distance_threshold) #create instance of the SORT tracker
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122, aspect='equal')
    track_id_list = []  #gnd

  for frame_idx, frame in tqdm(enumerate(sequence_segments.item().values())):  
  # for frame_idx, frame in enumerate(sequence_segments.item().values()):  
    if frame != []:
      clusters = [instance['points'] for instance in frame['seg instances'].values()]
      class_ids = [instance['class_ID'] for instance in frame['seg instances'].values()]
      track_ids = [instance['track_id'] for instance in frame['seg instances'].values()]
  
      points = np.zeros((1, 6))
      for idx, cluster in enumerate(clusters): 
        class_id = class_ids[idx]
        cluster = cluster.numpy().squeeze(axis=0)
        num_row = cluster.shape[0]
        class_id_vec = np.ones((num_row, 1)) * class_id  # 给每个tracker一个class——id 不变状态，只对class_id相同的进行association,class_id不同的矩阵对应位置赋为无穷大 
        cluster_with_class = np.concatenate((cluster, class_id_vec), axis=1)
        points = np.concatenate((points, cluster_with_class), axis=0)
      points = np.delete(points, 0, axis=0)
    
      if(display):
        # display gnd instances with scatter plot in ego-vehicle coordinate        
        for cluster_id, cluster in enumerate(clusters):
          track_id_array = track_ids[cluster_id]
          for i in range(cluster.shape[1]):
            y = cluster[:, i, 0]  # x_cc
            x = cluster[:, i, 1]  # y_cc
            try:
              track_id = track_id_array[i]
            except IndexError:
              track_id = track_id_array.item()
            if track_id not in track_id_list:
              color = COLOR[(len(track_id_list)-1)%NUM_COLOR] # new color
              track_id_list.append(track_id)
            else:
              color = COLOR[track_id_list.index(track_id)%NUM_COLOR]
            ax1.scatter(x, y, c=color, s=7)  # 
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)
        ax1.set_title('Ground Truth Tracks') # it is wrong
        # add another subplot for segmentation
    else:
      points = np.array([[1e4, 1e4, 1e4, 1e4, 1e4]])

      if(display):
        # empty frame
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)        
 

    start_time = time.time()
    tracked_instances = mot_tracker.update(frame)
    cycle_time = time.time() - start_time
    total_time += cycle_time

    for tracked_instance, tracker_id in tracked_instances:
      if(display):
        tracked_points = tracked_instance['points']
        color = COLOR[tracker_id%NUM_COLOR]
        ax2.scatter(tracked_points[:, :, 1], tracked_points[:, :, 0], c=color, s=7)
    ax2.set_xlabel('y_cc/m')
    ax2.set_xlim(50, -50)
    ax2.set_ylim(0, 100)
    ax2.set_title('Tracking')

    fig.canvas.flush_events()
    plt.show() 
    plt.pause(.1)
    if args.verbose:
      input("Press Enter to Continue")
    ax1.cla()
    ax2.cla()

