# Radar_PointNet_Panoptic_Segmentation
This repo is regarding the radar-based panoptic segmentation and tracking which is part of the radar-based data processing for ADAS and AD collaboration project. The main contributors are Jianan Liu, Shizhan Wang. The purpose of this project is to investigate algorithms of radar-based panoptic segmentation and tracking in BEV/3D space.


### NuScenes_Panoptic_Challenge
Lidar based Panoptic Segmentation and Tracking Challenge by using NuScenes Dataset

https://www.nuscenes.org/panoptic


Reference Paper: 

Panoptic Segmentation and Tracking
2021.Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking


Panoptic Tracking
2020.MOPT: Multi-Object Panoptic Tracking

https://arxiv.org/pdf/2004.08189.pdf





### Project9. Automotive Radar Point Cloud based Panoptic Segmentation and Tracking
a). 在现在的directly points processing deep learning network模型上，研究是否可能实现multi-task learning的方案，例如用同一个网络构架同时实现语义分割，实例分割，跟踪等多种任务（即基于radar detection points全景分割和全景跟踪？）。另外，（作为扩展内容）在这个基础之上，是否可以实现lane estimation以及drivable area estimation？

那么就需要思考并调研以下几个问题：

--Radar panoptic segmentation and tracking用radarscenes数据集可以做吗（换言之，radarscenes提供的panoptic segmentation and tracking的groundtruth信息吗）？只能做panoptic segmentation还是也可以做panoptic tracking？radar panoptic segmentation是否就是可以用multi-task learning的方案（一个task做dynamic points和static points的分割，另一个task做instance segmentation）直接做？又是否可能同时做3个task结合（一个task做dynamic points和static points的segmentation，另一个task做instance segmentation，第三个task做ghost/anomaly points分割）？

--如何利用静态点，（甚至在没有groundtruth的条件下）通过某些方式的deep learning实现lane estimation以及drivable area estimation？


参考文献：
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


具体方案如下：

a.1). 全景分割怎么做？直接用PointNet++网络接上不同的Head，对于所有点都做semantic segmentation，区分出static detection points和dynamic detection points之后，只对于dynamic detection points做instance segmentation。

a.2). 全景跟踪怎么做？直接用PointNet++网络，对于dynamic detection points：
a.2.1). 方案一：用网络估计多帧上点之间的similarity。(N * T) * (N * T)的similarity matrix，N为点数量，T为时间段长度。T可以选择比如3,然后通过(1,2,3),(2,3,4),(3,4,5)这种方案去估计连续时间上的similarity。
a.2.2). 方案二：用网络只估计单帧内的点之间的similarity，多帧之间用点之间的cosine similarity or normalized inner product的负数或者倒数作为cost，利用GNN和匈牙利算法估计帧间points间的association relation从而得到tracking。


**关于全景跟踪具体怎么实现的方案，可以参考这篇论文Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark
https://arxiv.org/pdf/2105.02440.pdf
这篇paper设计的网络以及data association的方案就类似于一堆点和另一堆点进行data association，与我们的全景跟踪场景非常相似。值得参考学习。**
