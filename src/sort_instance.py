"""
    SORT adpted version for tracking by instance segmentation
"""
from __future__ import print_function

import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from tqdm import tqdm
import time
import argparse
from filterpy.kalman import KalmanFilter
from sklearn import cluster

from distance import convex_hull_iou, euclidea_distance

np.random.seed(0)

# visualization for 5 classes
COLOR = ("red","green","black","orange","purple", "blue", "yellow", "cyan", "magenta")
CLASS = ("Car", "Pedestrian", "Pedestrian Group", "Two Wheeler", "Large Vehicle")


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(segment_instances, tracked_instances):
  """
  From SORT: Computes IOU between two instances in the form [x1,y1,x2,y2]
  return IOU matrix, the negative of cost matrix
  """
  num_seg = len(segment_instances) 
  num_track = len(tracked_instances)
  iou_matrix = np.zeros((num_seg, num_track))
  for row in range(num_seg):
    for col in range(num_track):
      iou_matrix[row, col] = convex_hull_iou(segment_instances[row], tracked_instances[col])
  return iou_matrix


def convert_centroid_to_instance(state:np.ndarray, frame:dict)->np.ndarray:
  '''
  Given updated state of instance centeroid, find the closest segemented instance
  '''
  pred_x, pred_y = state[:2]
  min_distance = 1e5
  optimal_instance = None
  for instance in frame.values():
    cluster = instance['points']
    segement_centroid_x, segement_centroid_y = extract_instance_centeroid(cluster)
    distance = euclidea_distance(segement_centroid_x, segement_centroid_y, 
                                  pred_x, pred_y)
    if min_distance > distance:
      min_distance = distance
      optimal_instance = instance # points with class ID
  return optimal_instance


def extract_instance_centeroid(instance:np.ndarray)->Tuple[float, float]:
  '''
  extract the centeroid of an instance by getting the mean of x, y
  '''
  x_center = instance[:, :, 0].mean()
  y_center = instance[:, :, 1].mean()
  return np.array((x_center, y_center)).reshape((2, 1))


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


    self.kf.x[:2] = extract_instance_centeroid(cluster)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,cluster):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(extract_instance_centeroid(cluster))

  def predict(self, frame):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if((self.kf.x[6]+self.kf.x[2])<=0):   
    #   self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1 # the number of prediction without update by new measurements
    self.history.append(convert_centroid_to_instance(self.kf.x, frame))
    return self.history[-1]

  def get_state(self, frame):
    """
    Returns the current instance estimate.
    """
    return convert_centroid_to_instance(self.kf.x, frame)


def associate_detections_to_trackers(instances, trackers, iou_threshold = 0.3):
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


def visualize_tracker():
  return


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, clusters:List[np.ndarray], frame):
    """
    Params:
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
      pred = self.trackers[t].predict(frame)
      trks.append(pred)
    #   if np.any(np.isnan(pos)):   # if any of the predictions is Nan, delete the tracker #？？？？ 
    #     to_del.append(t)
    # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    # for t in reversed(to_del):
    #   self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(clusters,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(clusters[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(clusters[i])   # one new tracker for each unmatched cluster   
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state(frame)
        # rule-based track management
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(d)
          # d:the current instance estimate  # 格式必须修改! 无法concatenate
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
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0

  if not os.path.exists('output'):
    os.makedirs('output')
  # load segments
  segments_path = os.path.join(args.seq_path, phase, 'SegmentSeq109.npy')
  sequence_segments = np.load(segments_path, allow_pickle='TRUE')
  
  mot_tracker = Sort(max_age=args.max_age, 
                      min_hits=args.min_hits,
                      iou_threshold=args.iou_threshold) #create instance of the SORT tracker
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    # fig.axis('off')
  for frame_idx, frame in tqdm(enumerate(sequence_segments.item().values())):  
    if frame != []:
      clusters = [instance['points'] for instance in frame.values()]
      class_ids = [instance['class_ID'] for instance in frame.values()]
  
    # start_time = time.time()
    # trackers = mot_tracker.update(clusters, frame)
    # cycle_time = time.time() - start_time
    # total_time += cycle_time
    # 空的frame该怎么办？

      if(display):
        # display segementor output with scatter plot in ego-vehicle coordinate
        for idx, cluster in enumerate(clusters):
          y = cluster[:, :, 0]  # x_cc
          x = cluster[:, :, 1]  # y_cc
          ax1.scatter(x, y, c=COLOR[idx%9], s=7)
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)
        fig.canvas.flush_events()
        plt.show() 
        plt.pause(.1)
        # input("Press Enter to Continue")
        ax1.cla()
    else:
      if(display):
        # display segementor output with scatter plot in ego-vehicle coordinat
        ax1.set_xlabel('x/m')
        ax1.set_ylabel('y/m')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(0, 100)
        # #plt.legend
        # 然后用凸包的线表示tracking结果
        # fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show() 
        plt.pause(.1)
        # input("Press Enter to Continue")
        ax1.cla()
      pass

  # print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, frame+1, (frame+1) / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
