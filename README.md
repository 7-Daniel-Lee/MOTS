# Radar_PointNet_Panoptic_Tracking_and_Segmentation
This repo is regarding the radar-based panoptic segmentation and tracking which is part of the radar-based data processing for ADAS and AD collaboration project. The purpose of this project is to investigate algorithms of radar-based panoptic segmentation and tracking in BEV/3D space.


## NuScenes_Panoptic_Challenge
Lidar based Panoptic Segmentation and Tracking Challenge by using NuScenes Dataset

https://www.nuscenes.org/panoptic


Reference Paper: 

#### Panoptic Segmentation and Tracking
2021.Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking

#### Panoptic Segmentation
2021(CVPR).4D Panoptic LiDAR Segmentation


#### Panoptic Tracking
2020(CVPR).MOPT: Multi-Object Panoptic Tracking

https://arxiv.org/pdf/2004.08189.pdf

2022(IROS).Learning Moving-Object Tracking with FMCW LiDAR

https://arxiv.org/pdf/2203.00959.pdf





## Project9. Automotive Radar Point Cloud based Panoptic Segmentation and Tracking（预计2022年4-6月主要阅读论文，理解设计idea，7至8月底前完成代码论文初稿）
* 注意：已经有人在RadarConf 2022上做过4D Imaging Radar Point Cloud based Panoptic Segmentation，见：
2022.Panoptic Segmentation for Automotive Radar Point Cloud
https://ieeexplore.ieee.org/document/9764218
不过还没有实现Panoptic Tracking

a). 在现在的directly points processing deep learning network模型上，研究是否可能实现multi-task learning的方案，例如用同一个网络构架同时实现语义分割，实例分割，跟踪等多种任务（即基于radar detection points全景分割和全景跟踪？）。另外，（作为扩展内容）在这个基础之上，是否可以实现lane estimation以及drivable area estimation？

那么就需要思考并调研以下几个问题：

--Radar panoptic segmentation and tracking用radarscenes数据集可以做吗（换言之，radarscenes提供的panoptic segmentation and tracking的groundtruth信息吗）？只能做panoptic segmentation还是也可以做panoptic tracking？radar panoptic segmentation是否就是可以用multi-task learning的方案（一个task做dynamic points和static points的分割，另一个task做instance segmentation）直接做？又是否可能同时做3个task结合（一个task做dynamic points和static points的segmentation，另一个task做instance segmentation，第三个task做ghost/anomaly points分割）？

--如何利用静态点，（甚至在没有groundtruth的条件下）通过某些方式的deep learning实现lane estimation以及drivable area estimation？


### 参考文献：
2021.MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning using an Anchor Free Approach
另外可以参考：
https://mp.weixin.qq.com/s/iYQkwCs1xz2QIjRkg5nerg
https://mp.weixin.qq.com/s/toAZS0OHdW4MG30P1wAAUA

Panoptic Segmentation and Tracking：
2021.Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking
https://arxiv.org/abs/2109.03805

Panoptic Tracking：
2020.MOPT: Multi-Object Panoptic Tracking
https://arxiv.org/pdf/2004.08189.pdf

With Transformer:
2022.Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers
https://arxiv.org/pdf/2109.03814.pdf


### 具体方案如下：

a.1). 全景分割怎么做？直接用PointNet++网络接上不同的Head，对于所有点都做semantic segmentation，区分出static detection points和dynamic detection points之后，只对于dynamic detection points做instance segmentation。

a.2). 全景跟踪怎么做？直接用PointNet++网络，对于dynamic detection points：
a.2.1). 方案一：用网络估计多帧上点之间的similarity。(N * T) * (N * T)的similarity matrix，N为点数量，T为时间段长度。T可以选择比如3,然后通过(1,2,3),(2,3,4),(3,4,5)这种方案去估计连续时间上的similarity。
a.2.2). 方案二：用网络只估计单帧内的点之间的similarity，多帧之间用点之间的cosine similarity or normalized inner product的负数或者倒数作为cost，利用GNN和匈牙利算法估计帧间points间的association relation从而得到tracking。

**关于全景跟踪具体怎么实现的方案，可以参考这篇论文Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark
https://arxiv.org/pdf/2105.02440.pdf
这篇paper设计的网络以及data association的方案就类似于一堆点和另一堆点进行data association，与我们的全景跟踪场景非常相似。值得参考学习。**

a.3). lane estimation以及drivable area estimation如何在可能没有groundtruth信息的条件下，通过(unsupervised/self-supervised )deep learning的方式来对静态点实现lane estimation以及drivable area estimation。

**注意：仍然需要探索结合Transformer或者visual MLP的方案。可以试着结合VAN（2022.Visual Attention Network  https://arxiv.org/abs/2202.09741 ）**





### 最新计划

1). Instace level tracking, evaluate using point-wise panoptic tracking metrics. 根据上一帧估计出的instance的predicted state和这一帧估计出的instance进行instance level的用匈牙利算法的GNN data association，这样其实就是只对instance tracking，只是最后把结果体现在point level上算metric即可。这种方案逻辑上不会修正instance segmentation的结果，panoptic tracking的效果极大的正比于instance segmentation准确度

2). point level tracking, evaluate using point-wise panoptic tracking metrics. 根据上一帧的points的predicted states和这一帧的points进行point level的用匈牙利算法的GNN data association（对于每个point都做tracking），这样得出上一帧和这一帧point间的对应关系，然后再和instance segmentation的结果比对，根据一些规则修正panoptic tracking的结果。这种方案逻辑上会修正instance segmentation的结果，panoptic tracking的output会对instance segmentation有个反馈，从而可以相互修正准确度

3). 以上1). 2). 都是用匈牙利算法实现GNN for data association + rule based track management，可是对于radar数据很可能用JPDA + rule based track management或者JIPDA效果会更好。所以可以考虑用JPDA + rule based track management或者JIPDA替换1). 2).中的GNN for data association + rule based track management。

* 关于用Kalman filter时候的速度状态估计：a). 速度作为一个被估计的状态，直接用Kalman filter估计速度。 b). 直接用measured Doppler径向速度作为估计的速度。 c). 用Kalman filter估计法向速度，measured Doppler作为径向速度，平方和开根作为估计的速度。

* 关于instance level data association的可能方案：a). 用Cluster的中心点和predicted track中心点的欧式距离做cost做GNN。 b). clusters分成小于2个点和多于2个点的，对于多于2个点的用IOU做cost先做GNN，然后再对小于2个点的用欧式距离再做一次GNN。


SORT
=====

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://alex.bewley.ai/misc/SORT-MOT17-06-FRCNN.webm).

By Alex Bewley  

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences in the [benchmark format](https://motchallenge.net/instructions/). To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

**Also see:**
A new and improved version of SORT with a Deep Association Metric implemented in tensorflow is available at [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort) .

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@bewley.ai).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


### Dependencies:

To install required dependencies run:
```
$ pip install -r requirements.txt
```


### Demo:

To run the tracker with the provided detections:

```
$ cd path/to/sort
$ python sort.py
```

To display the results you need to:

1. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```


### Main Results

Using the [MOT challenge devkit](https://motchallenge.net/devkit/) the method produces the following results (as described in the paper).

 Sequence       | Rcll | Prcn |  FAR | GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
--------------- |:----:|:----:|:----:|:-------------:|:-------------------:|:------------------:
 TUD-Campus     | 68.5 | 94.3 | 0.21 |  8   6   2   0|   15   113    6    9|  62.7  73.7  64.1
 ETH-Sunnyday   | 77.5 | 81.9 | 0.90 | 30  11  16   3|  319   418   22   54|  59.1  74.4  60.3
 ETH-Pedcross2  | 51.9 | 90.8 | 0.39 | 133  17  60  56|  330  3014   77  103|  45.4  74.8  46.6
 ADL-Rundle-8   | 44.3 | 75.8 | 1.47 | 28   6  16   6|  959  3781  103  211|  28.6  71.1  30.1
 Venice-2       | 42.5 | 64.8 | 2.75 | 26   7   9  10| 1650  4109   57  106|  18.6  73.4  19.3
 KITTI-17       | 67.1 | 92.3 | 0.26 |  9   1   8   0|   38   225    9   16|  60.2  72.3  61.3
 *Overall*      | 49.5 | 77.5 | 1.24 | 234  48 111  75| 3311 11660  274  499|  34.0  73.3  35.1


### Using SORT in your own project

Below is the gist of how to instantiate and update SORT. See the ['__main__'](https://github.com/abewley/sort/blob/master/sort.py#L239) section of [sort.py](https://github.com/abewley/sort/blob/master/sort.py#L239) for a complete example.
    
    from sort import *
    
    #create instance of SORT
    mot_tracker = Sort() 
    
    # get detections
    ...
    
    # update SORT
    track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...
    
 
How to save det.txt?  
[Format](https://github.com/abewley/sort/issues/61#issuecomment-497614711)
