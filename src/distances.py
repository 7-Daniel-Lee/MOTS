'''
Different distances for generating the negative cost matrix
'''
from numpy import ndarray, sqrt
import shapely
from shapely.geometry import Polygon,MultiPoint

def convex_hull_iou(gnd_instance:ndarray, seg_instance:ndarray)->float:
    '''
    get the minimum convex hull of two clusters, calculate their IoU
    param: gnd_instance: ground truth clusters with N rows, each row <x,y, rcs, vr>
           gnd_instance: segment clusters with M rows, each row <x,y, rcs, vr>
    return: convex hull iou
    '''
    # at least 3 points to be a convex hull
    assert (gnd_instance.shape[1] >= 3) and  (seg_instance.shape[1] >= 3)
           
    poly1 = Polygon(gnd_instance[:, :, :2].squeeze()).convex_hull 
    poly2 = Polygon(seg_instance[:, :, :2].squeeze()).convex_hull
    if not poly1.intersects(poly2): 
       return 0
    else:
       try:
              inter_area = poly1.intersection(poly2).area   
              union_area = poly1.area + poly2.area - inter_area
              if union_area == 0:
                     return 0
              iou=float(inter_area) / union_area
       except shapely.geos.TopologicalError:
              print('shapely.geos.TopologicalError occured, iou set to 0')
              return 0
    return iou


def cluster_iou(gnd_instance:ndarray, seg_instance:ndarray)->float:
    num_target = gnd_instance.shape[1]
    num_pred = seg_instance.shape[1]
    # ndarray ==> set of tuple
    
    target_set = set()
    for i in range(num_target):
        x = gnd_instance[:, i, 0][0]
        y = gnd_instance[:, i, 1][0]
        target_set.add((x, y))
        
    pred_set = set()
    for i in range(num_pred):
        x = seg_instance[:, i, 0][0]
        y = seg_instance[:, i, 1][0]
        pred_set.add((x, y))
    
    num_intersect = len(target_set.intersection(pred_set))
    num_union = len(target_set.union(pred_set)) + 1e-6  # avoid divided by 0 
    return num_intersect / num_union

def euclidean_distance(center1_x, center1_y, cetner2_x, center2_y)->float:
       return sqrt((center1_x-cetner2_x)*(center1_x-cetner2_x)+(center1_y-center2_y)*(center1_y-center2_y))