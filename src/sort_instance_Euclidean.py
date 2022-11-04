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

import torch
import argparse
from filterpy.kalman import KalmanFilter
from sklearn import cluster
from color_scheme import COLOR
from scipy.optimize import linear_sum_assignment
from distances import euclidean_distance

np.random.seed(0)

NUM_COLOR = len(COLOR)


def linear_assignment(cost_matrix:np.ndarray)->np.ndarray:
  '''
  Hungarian algorithm to solve the MOT association problem
  param: cost_matrix: row=number of gnd clusters in frame t
  col=number of prediction in frame t
  each element is the distance between a gnd cluster and a predicted cluster
  '''
  x, y = linear_sum_assignment(cost_matrix)
  return np.array(list(zip(x, y)))


def get_cost(segment_instances:List, tracked_instances:List)->np.ndarray:
  """
  From SORT: Computes IOU between two instances 
  param: segment_instances: gnd clusters in frame t, [cluster0, cluster1, ...] 
                            each cluster is a n*4*1 ndarray, each row is a point 
         tracked_instances: predicted clusters in frame t
  return: cost matrix
  """
  num_seg = len(segment_instances) 
  num_track = len(tracked_instances)
  cost_matrix = np.zeros((num_seg, num_track))
  for row in range(num_seg):
    for col in range(num_track):
      segment_x, segment_y = get_cluster_centeroid(segment_instances[row])
      tracked_x,  tracked_y = get_cluster_centeroid(tracked_instances[col])
      distance = euclidean_distance(segment_x, segment_y, tracked_x, tracked_y)
      cost_matrix[row, col] = distance
  return cost_matrix


def find_instance_from_prediction(pred:np.ndarray, frame:dict)->dict:
  '''
  Given predicted state of instance centeroid, find the closest segemented cluster
  param: pred: 
  return: optimal_instance: {'class_ID': 0-5, 'points':cluster, 'track_id': n}
  '''
  # empty frame
  if frame == []:
    return 
  pred_x, pred_y = pred[:2]
  min_distance = 1e5
  optimal_instance = None
  for instance in frame.values():
    cluster = instance['points']
    segement_centroid_x, segement_centroid_y = get_cluster_centeroid(cluster)
    distance = euclidean_distance(segement_centroid_x, segement_centroid_y, 
                                  pred_x, pred_y)
    if min_distance > distance:
      min_distance = distance
      optimal_instance = instance # points with class ID
  return optimal_instance


def get_cluster_centeroid(cluster:np.ndarray)->Tuple[float, float]:
  '''
  extract the centeroid of an cluster by getting the mean of x, y
  param: cluster: 1*n*4 each row is a point
  return: mass center of all the points
  '''
  x_center = cluster[:, :, 0].mean()
  y_center = cluster[:, :, 1].mean()
  return np.array((x_center, y_center)).reshape((2, 1))


def get_mean_doppler_velocity(cluster:np.ndarray)->Tuple[float, float]:
  '''
  1. decompose each Vr along the x and y axis 2. get the mean Vrx, Vry
  Note: the mean Vr is no longer radial of the mass center
  '''
  xcc = cluster[:, :, 0]
  ycc = cluster[:, :, 1]
  vr = cluster[:, :, 2]
  cosine_thetas =  xcc / np.sqrt(xcc*xcc + ycc*ycc)
  sine_thetas = ycc / np.sqrt(xcc*xcc + ycc*ycc)
  vr_x = np.squeeze(vr*cosine_thetas)
  vr_y = vr*sine_thetas
  mean_vr_x = np.mean(vr_x)
  mean_vr_y = np.mean(vr_y)
  return np.array((mean_vr_x, mean_vr_y)).reshape((2, 1))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,cluster):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2) 
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]) # 改成deta t 反应真实速度
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
    # H is measurement matrix which represents the measurement model in update step.
    # Motion noise covariance matrix, Q
    # Measurement noise covariance matrix, R
    # State covariance matrix, P

  # self.kf.R[2:,2:] *= 10.  #?  corresponds to s, r #打断点看一下值
    self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # in this way, we can choose if we trust measurement!!!!
    self.kf.P *= 10.   # initial value
    self.kf.Q[2:, 2:] *= 0.01  # 通过折半查找寻优


    self.kf.x[:2] = get_cluster_centeroid(cluster)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, associated_cluster):
    """
    Updates the state vector with measurements
    param: associated_cluster: found cluster that is closest to the prediction state
    return: Kalman filter's state = associated measurement
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.x[:2] = get_cluster_centeroid(associated_cluster)  
    self.kf.x[2:] = get_mean_doppler_velocity(associated_cluster) 
    
  def predict(self, frame):
    """
    Advances the state vector and returns the predicted instance
    param: frame: frame at t+1
    return: a instance, dictionary {class_id:xxx, points: ndarray}
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1 # the number of prediction without update by new measurements, set 0 if updated
    self.history.append(find_instance_from_prediction(self.kf.x, frame))  
    return self.history[-1] # only return the latest prediction

  def get_state(self, frame):
    """
    Returns the current instance estimate.
    """
    return find_instance_from_prediction(self.kf.x, frame)


def associate_detections_to_trackers(instances:List, trackers:List, distance_threshold = 0.3):  # tune threshhold！
  """
  Assigns instance segmentations to tracked object (both represented as clusters)
  param trackers: trackers' predictions in this frame

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(instances)), np.empty((0,5),dtype=int)  # change!  # 为什么中间那个是arange，其他两个是empty？
    # matched, unmatched_dets, unmatched_trks

  cost_matrix = get_cost(instances, trackers)

  if min(cost_matrix.shape) > 0:
    a = (cost_matrix < distance_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1) # coordinates in the cost matrix
    else:
      matched_indices = linear_assignment(cost_matrix) # 
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
    if(cost_matrix[m[0], m[1]]>distance_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
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
    if frame == []:
      return
    # get predicted locations from existing trackers, associate the detections with predictions  
    trks = []  
    # to_del = []
    ret = []
    for t in range(len(self.trackers)):
      pred = self.trackers[t].predict(frame)['points'] # ndarray  主要问题是1312´帧是空的!!
      trks.append(pred)
    #   if np.any(np.isnan(pred.shape)):   # if any of the predictions is Nan, delete the tracker 似乎是多余的？因为只要frame不空，一定能返回一个cluster
    #     to_del.append(t)
    # for t in reversed(to_del):
    #   self.trackers.pop(t)

    # if frame == []:
    #   clusters = [torch.from_numpy(np.array([[[1e4, 1e4, 1e4, 1e4]]]))]
    # else:
    clusters = [instance['points'] for instance in frame.values()] # a list of ndarray

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(clusters, trks, self.distance_threshold)
    # 可以分别做2次association,小于两个点的用距离，大于的用IOU

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(clusters[m[0]])   # m is the index for matched clusters

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


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('-d', '--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument('-s', '--save', dest='save', help='Save each frame of the animation', action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data_short')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=3)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=0)
    parser.add_argument("--distance_threshold", help="Minimum IOU for match.", type=float, default=0.1)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()
    return args


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
  segments_path = os.path.join(args.seq_path, phase, 'Seq109_gnd&seg.npy')
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
      pred_clusters = [instance['points'] for instance in frame['seg instances'].values()]
      class_ids = [instance['class_ID'] for instance in frame['seg instances'].values()]
      gnd_clusters = [instance for instance in frame['gnd instances'].values()]
      gnd_track_ids = [track_id for track_id in frame['gnd instances'].keys()] # N_cluster * 1
      
      if(display):
        # display gnd instances with scatter plot in ego-vehicle coordinate        
        for cluster_id, cluster in enumerate(gnd_clusters):
          track_id = gnd_track_ids[cluster_id]
          X = cluster[:, :, 1]
          Y = cluster[:, :, 0]
          if track_id not in track_id_list:
            color = COLOR[(len(track_id_list)-1)%NUM_COLOR] # new color
            track_id_list.append(track_id)
          else:
            color = COLOR[track_id_list.index(track_id)%NUM_COLOR]
          ax1.scatter(X, Y, c=color, s=7)  
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)
        ax1.set_title('Segmentation')
    else:
      # points = np.array([[1e4, 1e4, 1e4, 1e4, 1e4]])

      if(display):
        # empty frame
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)   
 
    # associate instance segmentation
    start_time = time.time()
    tracked_instances = mot_tracker.update(frame['seg instances'])
    cycle_time = time.time() - start_time
    total_time += cycle_time

    if(display):
      for tracked_instance, tracker_id in tracked_instances:
        tracked_points = tracked_instance['points']
        color = COLOR[tracker_id%NUM_COLOR]  # the same tracker_id uses the same color
        ax2.scatter(tracked_points[:, :, 1], tracked_points[:, :, 0], c=color, s=7)
      ax2.set_xlabel('y_cc/m')
      ax2.set_xlim(50, -50)
      ax2.set_ylim(0, 100)
      ax2.set_title('Tracking')  
      
      if args.save: # save images
        plt.savefig('img/{}.jpg'.format(frame_idx))

      fig.canvas.flush_events()
      plt.show() 
      plt.pause(.1)
      if args.verbose:
        input("Press Enter to Continue")
      ax1.cla()
      ax2.cla()

# export PYTHONPATH=.
# python src/sort_instance_Euclidean.py -v