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
    # 如果只有一个点就用正负1得到一个很小的凸包，合适吗？
    poly1 = Polygon(gnd_instance[:, :, :2].squeeze()).convex_hull # 如果只有一个点无法形成凸包？？？
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


def euclidean_distance(center1_x, center1_y, cetner2_x, center2_y)->float:
       return sqrt((center1_x-cetner2_x)*(center1_x-cetner2_x)+(center1_y-center2_y)*(center1_y-center2_y))