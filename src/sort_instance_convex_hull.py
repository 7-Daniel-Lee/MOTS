"""
    SORT adpted version for tracking by instance segmentation
"""
from __future__ import print_function
from cmath import nan

import os
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
from distance import convex_hull_iou
from src.sort_instance_Euclidean import find_instance_from_prediction, get_cluster_centeroid, get_mean_doppler_velocity, KalmanBoxTracker, Sort, parse_args


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
  iou_matrix = np.zeros((num_seg, num_track))
  for row in range(num_seg):
    for col in range(num_track):
      iou_matrix[row, col] = convex_hull_iou(segment_instances[row].numpy(), tracked_instances[col].numpy())
  return iou_matrix


def associate_detections_to_trackers(instances:List, trackers:List, iou_threshold = 0.3):
  """
  Assigns instance segmentations to tracked object (both represented as clusters)
  param trackers: trackers' predictions in this frame

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(instances)), np.empty((0,5),dtype=int)  # change!  # 为什么中间那个是arange，其他两个是empty？
    # matched, unmatched_dets, unmatched_trks

  iou_matrix = iou_batch(instances, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(instances):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d) # list of instances id
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



if __name__ == '__main__':
  # all train
  args = parse_args()
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
    if frame != []:
      clusters = [instance['points'] for instance in frame.values()]
      class_ids = [instance['class_ID'] for instance in frame.values()]
      track_ids = [instance['track_id'] for instance in frame.values()]
  
      points = np.zeros((1, 6))
      for idx, cluster in enumerate(clusters): 
        class_id = class_ids[idx]
        # print(cluster.shape)
        cluster = cluster.numpy().squeeze(axis=0)
        # print(cluster.shape)
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
        ax1.set_title('Segmentation')
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
      # print('tracked instance')
      if(display):
        tracked_points = tracked_instance['points']
        color = COLOR[tracker_id%NUM_COLOR]
        ax2.scatter(tracked_points[:, :, 1], tracked_points[:, :, 0], c=color, s=7)
    ax2.set_xlabel('y_cc/m')
    # ax2.set_ylabel('x_cc/m')
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

