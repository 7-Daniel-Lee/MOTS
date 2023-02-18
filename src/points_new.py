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

def linear_assignment(cost_matrix):
  '''
  Hungarian algorithm to solve the MOT association problem
  '''
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
  
def iou_batch(points:np.ndarray,pred_points:np.array,association_flag:str)->np.ndarray:
  num_seg=points.shape[0]
  num_track=pred_points.shape[0]
  cost_matrix = np.zeros((num_seg, num_track))
  for row in range(num_seg):
    for col in range(num_track):
      if association_flag == 'Euclidean':
        segment_x, segment_y = points[row, 0:2]
        tracked_x, tracked_y = points[col, 0:2]
        distance = euclidean_distance(segment_x, segment_y, tracked_x, tracked_y)
        cost_matrix[row, col] = distance
      elif association_flag == 'IOU':
        cost_matrix[row, col] = -convex_hull_iou(points.shape[row], pred_points.shape[col])
  return cost_matrix

class KalmanBoxTracker(object):
  count = 0
  def __init__(self,point):
    self.kf = KalmanFilter(dim_x=4, dim_z=2)
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    self.kf.R[2:,2:] *= 10.
    self.kf.P[2:,2:] *= 1000.
    self.kf.P *= 1.
    self.kf.Q[2:, 2:] *= 0.01

    self.kf.x[:2] = point[0:2].reshape((2, 1))
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,point):
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(point[0:2])

  def predict(self):
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]
  
  def get_state(self):
    return self.kf.x
  

def associate_detections_to_trackers(points, trackers, distance_threshold = 0.3):
  if len((trackers) == 0):
    return np.empty((0,2),dtype=int), np.arange(len(points)),np.empty((0,5),dtype=int)
  
  cost_matrix = iou_batch(points, trackers)

  if min(cost_matrix.shape) > 0:
    a = (cost_matrix < distance_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(cost_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))
  
  unmatched_detections = []
  for d, det in enumerate(points):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

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
    self.max_age = max_age
    self.min_hits = min_hits
    self.distance_threshold = distance_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, points:np.ndarray):
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 4))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pred = self.trackers[t].predict()
      trk[:] = pred.squeeze()
      if np.any(np.isnan(pred.shape)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(points, trks, self.distance_threshold)
    for m in matched:
      self.trackers[m[1]].update(points[m[0], :])

    for i in unmatched_dets:
      trk = KalmanBoxTracker(points[i, :])
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state() 
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(d)
      i -= 1
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
    if(len(ret)>0):
      return ret
    return np.empty((0,5))
  
def parse_args():
  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument('-d', '--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
  parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
  parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='')
  parser.add_argument("--max_age",
                      help="Maximum number of frames to keep alive a track without associated detections.", 
                      type=int, default=3)
  parser.add_argument("--min_hits", 
                      help="Minimum number of associated detections before track is initialised.", 
                      type=int, default=2)
  parser.add_argument("--distance_threshold", help="Minimum IOU for match.", type=float, default=5)
  parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  if args.verbose:
    basicConfig(level=DEBUG)
  else:
    basicConfig(level=INFO)
  display = args.display
  display = True
  total_time = 0.0

  if not os.path.exists('output'):
    os.makedirs('output')
  segments_path = os.path.join(args.seq_path, args.phase, 'SegmentSeq109_trackid.npy')
  sequence_segments = np.load(segments_path, allow_pickle='TRUE')

  mot_tracker = Sort(max_age=args.max_age, 
                     min_hits=args.min_hits,
                     distance_threshold=args.distance_threshold) 
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122, aspect='equal')
    track_id_list = []

  for frame_idx, frame in tqdm(enumerate(sequence_segments.item().values())):
    if frame != []:
      clusters = [instance['points'] for instance in frame.values()]
      class_ids = [instance['class_ID'] for instance in frame.values()]
      track_ids = [instance['track_id'] for instance in frame.values()]

      points = np.zeros((1, 6))
      for idx, cluster in enumerate(clusters): 
        class_id = class_ids[idx]
        cluster = cluster.numpy().squeeze(axis=0)
        num_row = cluster.shape[0]
        class_id_vec = np.ones((num_row, 1)) * class_id
        cluster_with_class = np.concatenate((cluster, class_id_vec), axis=1)
        points = np.concatenate((points, cluster_with_class), axis=0)
      points = np.delete(points, 0, axis=0)


      if(display):
        for cluster_id, cluster in enumerate(clusters):
          track_id_array = track_ids[cluster_id]
          for i in range(cluster.shape[1]):
            y = cluster[:, i, 0] 
            x = cluster[:, i, 1] 
            try:
              track_id = track_id_array[i]
            except IndexError:
              track_id = track_id_array.item()
            if track_id not in track_id_list:
              color = COLOR[(len(track_id_list)-1)%NUM_COLOR]
              track_id_list.append(track_id)
            else:
              color = COLOR[track_id_list.index(track_id)%NUM_COLOR]
            ax1.scatter(x, y, c=color, s=7)
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)
        ax1.set_title('Segmentation')
    else:
      points = np.array([[1e4, 1e4, 1e4, 1e4, 1e4]])

      if(display):
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100) 
      start_time = time.time()
      tracked_points = mot_tracker.update(points)
      cycle_time = time.time() - start_time
      total_time += cycle_time
      
      
      for tracked_point in tracked_points:
        print('Tracked Points x:{} y:{}'.format(tracked_point[0], tracked_point[1]))
    if display:
      tracked_points = np.array(tracked_points)
      ax2.scatter(tracked_points[:, 1], tracked_points[:, 0], c='b', s=7)
      ax2.set_xlabel('y_cc/m')
      ax2.set_ylabel('x_cc/m')
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