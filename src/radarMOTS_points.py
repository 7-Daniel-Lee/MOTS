from __future__ import print_function
from cmath import nan

import os
from sys import displayhook
from turtle import distance
from typing import List, Tuple
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from itertools import cycle
from logging import basicConfig, DEBUG, INFO
from tqdm import tqdm
import time

import torch
import argparse
from filterpy.kalman import KalmanFilter
from sklearn import cluster
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from color_scheme import COLOR
from scipy.optimize import linear_sum_assignment
from distances import euclidean_distance
from distances import convex_hull_iou

np.random.seed(0)

NUM_COLOR = len(COLOR)

def linear_assignment(cost_matrix):
  '''
  Hungarian algorithm to solve the MOT association problem
  '''
  try:
    import lap
    from lap import lapjv
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
  

  
def iou_batch(points:np.ndarray,pred_points:np.array)->np.ndarray:
  'From SORT: Computes IOU between two instances'
  
  num_seg=points.shape[0]
  num_track=pred_points.shape[0]
  cost_matrix = np.zeros((num_seg, num_track))
  for row in range(num_seg):
    for col in range(num_track):
      segment_x, segment_y = points[row, 0:2]
      tracked_x, tracked_y = points[col, 0:2]
      distance = euclidean_distance(segment_x, segment_y, tracked_x, tracked_y)
      cost_matrix[row, col] = distance
  return cost_matrix


# def find_point_from_prediction(pred_points:np.array, frame:dict)->dict:
#   '''
#   Given predicted state of instance centeroid, find the closest segemented cluster
#   '''
#   if frame == []:  #如果此帧为空
#     return   
#   pred_x, pred_y = pred_points[0:2]
#   min_distance = 1e5
#   optimal_point = None
#   for instance in frame['seg instances'].values():
#     cluster = instance['points']
#     segement_centroid_x, segement_centroid_y = get_cluster_centeroid(cluster)
#     distance = euclidean_distance(segement_centroid_x, segement_centroid_y, 
#                                   pred_x, pred_y)
#     if min_distance > distance:
#       min_distance = distance
#       optimal_point = instance # points with class ID
#   return optimal_point


# def get_cluster_centeroid(cluster:np.ndarray)->Tuple[float, float]:
#   '''
#   extract the centeroid of an cluster by getting the mean of x, y
#   '''
#   x_center = cluster[:, :, 0].mean()
#   y_center = cluster[:, :, 1].mean()
#   return np.array((x_center, y_center)).reshape((2, 1))


class KalmanBoxTracker(object):
  count = 0
  def __init__(self,point):
    'Define the parameters of the Kalman filter'
    self.kf = KalmanFilter(dim_x=4, dim_z=2)  #因为输入数据有4个状态，所以dim_x=4,需要观测2个状态，所以dim_z=2
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    self.kf.R[2:,2:] *= 10.
    self.kf.P[2:,2:] *= 1000.  #give high uncertainty to the unobservable initial velocities
    # in this way, we can choose if we trust measurement
    self.kf.P *= 1.  # initial value
    self.kf.Q[2:, 2:] *= 0.01
    # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
    # H is measurement matrix which represents the measurement model in update step.
    # Motion noise covariance matrix, Q
    # Measurement noise covariance matrix, R
    # State covariance matrix, P

    self.kf.x[:2] = point[0:2].reshape((2, 1))
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
  

  def update(self, point):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0  # 需要在这里更改状态 self.kf.x的值，直接等于测量的质心和平均Vr  
    self.history = []
    self.hits += 1
    self.hit_streak += 1  # the total number of times it consecutively got matched with a detection in the last frames.
    # self.kf.update(point[0:2])
    self.kf.x[:2] = point[0:2]
    self.kf.x[2:] = point[2:]
    #此处更新直接赋予关联后的点的值

  def predict(self):
    'Advances the state vector and returns the predicted bounding box estimate.'
    'return: a instance, dictionary {class_id:xxx, points: ndarray}'
    
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1  # the number of prediction without update by new measurements
    self.history.append(self.kf.x)
    # self.history.append(find_point_from_prediction(self.kf.x, frame))
    return self.history[-1]  # only return the latest prediction
  
  def get_state(self):
    'Returns the current instance estimate.'
    return self.kf.x
    # return find_point_from_prediction(self.kf.x, frame)
  

def associate_detections_to_trackers(points, trackers, distance_threshold = 0.3):
  'Assigns instance segmentations to tracked object (both represented as clusters)'
  'param trackers: trackers predictions in this frame'
  'Returns 3 lists of matches, unmatched_detections and unmatched_trackers'
  if len((trackers) == 0) or len((points) == 0):    #如果跟踪器为空
    return np.empty((0,2),dtype=int), np.arange(len(points)),np.empty((0,5),dtype=int)
    # matched, unmatched_dets, unmatched_trks
  
  cost_matrix = iou_batch(points, trackers)

  if min(cost_matrix.shape) > 0:
    a = (cost_matrix < distance_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)  #堆叠作用，coordinates in the cost matrix
      #此处为特殊情况，即点在前后帧没有发生过移动
    else:
      matched_indices = linear_assignment(cost_matrix)  #利用匈牙利算法进行data association
  else:
    matched_indices = np.empty(shape=(0,2))
  
  # 记录未匹配的检测框及跟踪框
  # 未匹配的检测框放入unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标
  unmatched_detections = []
  for d, det in enumerate(points):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)  # list of instances id
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)     #如果跟踪器中第t个跟踪结果不在匹配结果索引中，则t未匹配上
    #filter out matched with low IOU


  matches = []    #存放过滤后的匹配结果
  for m in matched_indices:    #遍历粗匹配结果
    if(cost_matrix[m[0], m[1]]>distance_threshold):     #m[0]是检测器ID， m[1]是跟踪器ID，如它们的代价矩阵大于阈值则将它们视为未匹配成功
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))   #将过滤后的匹配对维度变形成1x2形式
  if(len(matches)==0):   #如果过滤后匹配结果为空，那么返回空的匹配结果
    # 初始化matches,以np.array的形式返回
    matches = np.empty((0,2),dtype=int)
  else:    #如果过滤后匹配结果非空，则按0轴方向继续添加匹配对
    matches = np.concatenate(matches,axis=0)
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
  

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, distance_threshold=0.3):
    'Sets key parameters for SORT'
    self.max_age = max_age
    self.min_hits = min_hits  #  the minimum value of hit streak of a track, required, such that it gets displayed in the outputs.
    self.distance_threshold = distance_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, points:np.ndarray):
    '''
    points: dictionary 
    clusters - a list of ndarray 
    this method must be called once for each points even with empty detections (use np.empty((0, 5)) for points without detections).
    Returns the a similar array, where the last column is the object ID.
    '''
    self.frame_count += 1  #帧计数
    #get predicted locations from existing trackers, associate the detections with predictions  
    trks = np.zeros((len(self.trackers), 4))   # 根据当前所有卡尔曼跟踪器的个数创建二维零矩阵
    to_del = []
    ret = []
    for t, trk in enumerate(trks):   #循环遍历卡尔曼跟踪器列表
      pred = self.trackers[t].predict()  # ndarray，用卡尔曼跟踪器t 预测对应物体在当前帧中的点
      trk[:] = pred.squeeze()
      if np.any(np.isnan(pred.shape)):  # if any of the predictions is Nan, delete the tracker
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  #Mask elements are supported in numeric arrays
    #将预测为空的卡尔曼跟踪器所在行删除，最后trks中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空的点
    #Mask invalid data in arrays, such as NaN or inf
    for t in reversed(to_del):  #倒序遍历待删除列表
      self.trackers.pop(t)   #从跟踪器中删除 to_del中的上一帧跟踪器ID

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(points, trks, self.distance_threshold)
    #对传入的检测结果 与 上一帧跟踪物体在当前帧中预测的结果做关联，返回匹配的目标矩阵matched, 新增目标的矩阵unmatched_dets, 离开画面的目标矩阵unmatched_trks
    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(points[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(points[i, :])  # one new tracker for each unmatched cluster，将新增的未匹配的检测结果dets[i,:]传入KalmanBoxTracker
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()  #获取trk跟踪器的状态
      # rule-based track management 持续更新+连续match数量大于最小阈值或者还没到更新次数还没达到该阈值,最初几帧
      # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
      #此处有问题：如果加上此if判断语句的话则无法呈现出第2帧及其之后的帧数，是否为初始参数max_age and min_hit设置的不正确？
      ret.append(d)  #添加到待保存列表
      i -= 1
      #delete death track
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
    if(len(ret)>0):
      return ret
    return np.empty((0,5))
  
def parse_args():
  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument('-a',
                        '--association',
                        help='association metric to generate the cost matrix, Euclidean distance',
                        type=str,
                        default='EuclideanDistance'
  )
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
  args.seq_path = 'C:\\Users\\12283\\Desktop\\SCI\\Radar_PointNet_Panoptic_Tracking_and_Segmentation-main\\Radar_PointNet_Panoptic_Tracking_and_Segmentation-main\\src\\data_short'
  if args.verbose:
    basicConfig(level=DEBUG)
  else:
    basicConfig(level=INFO)
  # display = args.display
  display = True
  total_time = 0.0

  if not os.path.exists('output'):
    os.makedirs('output')
  segments_path = os.path.join(args.seq_path, args.phase, 'SegmentSeq109_trackid.npy')
  sequence_segments = np.load(segments_path, allow_pickle='TRUE')

  mot_tracker = Sort(max_age=args.max_age, 
                     min_hits=args.min_hits,
                     distance_threshold=args.distance_threshold) #create instance of the SORT tracker
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122, aspect='equal')
    track_id_list = []

  for frame_idx, frame in tqdm(enumerate(sequence_segments.item().values())):  #从第0帧开始循环遍历每一帧的points,class_ID,tracked_id的值
    if frame != []:
      clusters = [instance['points'] for instance in frame['seg instances'].values()]
      class_ids = [instance['class_ID'] for instance in frame['seg instances'].values()]
      track_ids = [instance['track_id'] for instance in frame['seg instances'].values()]
      

      points = np.zeros((1, 6))
      for idx, cluster in enumerate(clusters): 
        class_id = class_ids[idx]
        # print(cluster.shape)
        cluster = cluster.numpy().squeeze(axis=0)
        # print(cluster.shape)
        num_row = cluster.shape[0]
        class_id_vec = np.ones((num_row, 1)) * class_id
        # 给每个tracker一个class—id 不变状态，只对class_id相同的进行association,class_id不同的矩阵对应位置赋为无穷大
        cluster_with_class = np.concatenate((cluster, class_id_vec), axis=1)  #在第1轴上重新组合
        points = np.concatenate((points, cluster_with_class), axis=0)  #在第0轴上重新组合
      points = np.delete(points, 0, axis=0)  #删去第0轴为0的值
      # points = np.squeeze(points)
      

      if(display):
        # display gnd instances with scatter plot in ego-vehicle coordinate
        for cluster_id, cluster in enumerate(clusters):
          track_id_array = track_ids[cluster_id]
          for i in range(cluster.shape[1]):
            y = cluster[:, i, 0] # x_cc
            x = cluster[:, i, 1] # y_cc
            try:
              track_id = track_id_array[i]
            except IndexError:
              track_id = track_id_array.item()
            if track_id not in track_id_list:
              color = COLOR[(len(track_id_list)-1)%NUM_COLOR]  # new color
              track_id_list.append(track_id)
            else:
              color = COLOR[track_id_list.index(track_id)%NUM_COLOR]
            ax1.scatter(x, y, c=color, s=7)
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100)
        ax1.set_title('Groudtruth')


      start_time = time.time()
      tracked_points = mot_tracker.update(points) #更新sort跟踪器
      cycle_time = time.time() - start_time  #sort跟踪器耗时
      total_time += cycle_time  #sort跟踪器总耗时

      for tracked_point in tracked_points:
        print('Tracked Points x:{} y:{}'.format(tracked_point[0], tracked_point[1]))
      if (display):
        tracked_points = np.array(tracked_points)
        ax2.scatter(tracked_points[:,1], tracked_points[:,0], c=color, s=7)
        ax2.set_xlabel('y_cc/m')
        ax2.set_ylabel('x_cc/m')
        ax2.set_xlim(50, -50)
        ax2.set_ylim(0, 100)
        ax2.set_title('Tracking')
    else:
      points = np.array([[1e4, 1e4, 1e4, 1e4, 1e4]])

      if(display):
        # empty frame
        ax1.set_xlabel('y_cc/m')
        ax1.set_ylabel('x_cc/m')
        ax1.set_xlim(50, -50)
        ax1.set_ylim(0, 100) 
      


    fig.canvas.flush_events()
    plt.show()
    plt.pause(.1)
    if args.verbose:
      input("Press Enter to Continue")
    ax1.cla()
    ax2.cla()
