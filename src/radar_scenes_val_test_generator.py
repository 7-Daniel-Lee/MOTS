'''
The objective of this set of utility functions is to generate processed radar data in order to
feed it into PointNet Semantic Segmentation networks.

In order to process the data, we deviced the following functions:
1. gen_timeline: from the json file, we read out the sequence_timeline and anchor_timeline.
2. features_from_radar_data: there are only 4/5 points out of radar_data are required for our purposes.
3. radar_scenes_dataset_partition: read out the json file for partitioning of the 158 sequences. 
4. get_valid_points/get_non_static_points: filter out invalid radar points and non_static points.
5. synchronize: based on the anchor reference, to convert global coordinate into a single ego_coordinate for all four radars. 
6. partitioned_data_generator: based on the sequence partition specified, generate data accordingly


For interfacing with torch.nn the DataLoader functin is utilized. First we have to define our
own datastructure with Dataset class and load it through DataLoader.
'''
import argparse
import os
import random
random.seed(0)
import pandas as pd
import h5py
import json
import numpy as np
from typing import Union
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
import torch

from instance_seg.radar_scenes_read_json import Sequence
from instance_seg.radar_scenes_labels import Label, ClassificationLabel 

class DataAugmentation(object):
    """
    add noise to original data
  
    """
    def __call__(self, sample):
        sample['x_cc'] = np.random.normal( sample['x_cc'], 0.1/3 ) # Perturb detection positions independently
        sample['y_cc'] = np.random.normal( sample['y_cc'], 0.1/3 )
        sample['x_cc'] += np.random.normal( 0, 10 )  # Shift whole point cloud position
        sample['y_cc'] += np.random.normal( 0, 10 )
        sample['vr_compensated'] = np.random.normal( sample['vr_compensated'], 0.2 ) # Perturb velocity
        sample['rcs'] = np.random.normal( sample['rcs'], 1 )#Perturb RCS

        return sample


class FeatureEngineering(object):
    def __init__(self):
        # Define a dictionary "aux_fun"(aux stands for auxiliary), the dictionary has several "key : value", e.g. 'radius' as key and the returned value from defined lambda function is value
        self.aux_fun = {
            # The lambda expression is a way to define the function. For example showing as below, "sample" is the input arguement/variable 
            # and the definition of function is "np.sqrt( sample[0]**2 + sample[1]**2" 
            'radius':   lambda sample: np.sqrt( sample['x_cc']**2 + sample['y_cc']**2 ),
            'angle':    lambda sample: np.arctan2( sample['x_cc'], sample['y_cc'] )
        }
        self.feature_fun = {
            # Statistical values for all the detection points in one frame(Here what we acutally mean is "statistical values for all the detection points 
            # belong to single object in one frame", due that we only have one moving object in FOV for the measured data in 20190730)
            'RCS_mean':     lambda sample,aux: sample['rcs'].mean(),
            'RCS_std':      lambda sample,aux: sample['rcs'].std(),
            'RCS_spread':   lambda sample,aux: sample['rcs'].max() - sample['rcs'].min(),
            # For the (ego-motion compensated)velocity v_x we also design feature "the fraction of targets with v_x < 0.3 m/s"
            'v_n_le_0p3':   lambda sample,aux: np.mean(sample['velx'] < 0.3 ),
            'range_min':    lambda sample,aux: aux['radius'].min(),
            'range_max':    lambda sample,aux: aux['radius'].max(),
            'range_mean':   lambda sample,aux: aux['radius'].max(),
            'range_std':    lambda sample,aux: aux['radius'].std(),
            'ang_spread':   lambda sample,aux: aux['angle'].max() - aux['angle'].min(),
            'ang_std':      lambda sample,aux: aux['angle'].std(),
            'hist_v':       lambda sample,aux: np.histogram( sample['vr_compensated'], bins=10)[0],
            'hist_RCS':     lambda sample,aux: np.histogram( sample['rcs'],  bins=10)[0],
            # The two eigenvalues of the covariance matrix of x and y, proposed in paper: "2018. Comparison of Random Forest and Long
            # Short-Term Memory Network Performances in Classification Tasks Using Radar". The eigenvalues represent the variance of 
            # the data along the eigenvector directions, so here we expect the random forest classifier should "learn" something related 
            # to "shape information of single object" 
            'eig_cov_xy':   lambda sample,aux: np.sort( np.linalg.eig( np.cov(np.vstack((sample[0],sample[1])),bias=True) )[0] )
        }
    def __call__(self, sample):
        """
            Read each "sample"(one "sample" has all attributes for all detection points in one frame), calculate features for the input frame
        """
        features = {}
        # Get a dictionary, aux, which contains values of 'radius' and 'angle' for each of all detection points in one frame
        aux  = { key:fun(sample) for key,fun in self.aux_fun.items() } # aux stands for auxiliary
        # For each feature function in "feature_fun dictionary"
        for featname, featfun in self.feature_fun.items():
            # Get the returned value of current feature function, featfun
            featvalue = featfun( sample, aux )
            # If the feature function, featfun, returns an array, then we need to store each position in separate keys
            if type(featvalue) is np.ndarray:
                for i in range(len(featvalue)):
                    featname_i = featname+'_'+str(i)
                    features[featname_i] = np.append(features[featname_i],featvalue[i]) if featname_i in features else featvalue[i]
            else:
                features[featname] = np.append(features[featname],featvalue) if featname in features else featvalue
        # Convert to dataframe
        framenumber = sample.name
        return pd.Series(features, name=framenumber)


def batch_transform_3d_vector(trafo_matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 transformation matrix to every (1,3) vector contained in vec.
    Vec has shape (n_vec, 3)
    :param trafo_matrix: numpy array with shape (3,3)
    :param vec: numpy array with shape (n_vec, 3)
    :return: Transformed vector. Numpy array of shape (n_vec, 3)
    """
    return np.einsum('ij,kj->ki', trafo_matrix, vec)


def trafo_matrix_seq_to_car(odometry: np.ndarray) -> np.ndarray:
    """
    Computes the transformation matrix from sequence coordinates to car coordiantes, given an odometry entry.
    :param odometry: Numpy array containing at least the names fields "x_seq", "y_seq" and "yaw_seq" which give the
    position and orientation of the sensor vehicle.
    :return: Numpy array with shape (3,3), the transformation matrix. Last column is the translation vector.
    """
    x_car = odometry["x_seq"]
    y_car = odometry["y_seq"]
    yaw_car = odometry["yaw_seq"]
    c = np.cos(yaw_car)
    s = np.sin(yaw_car)
    return np.array([[c, s, -x_car * c - y_car * s],
                     [-s, c, x_car * s - y_car * c],
                     [0, 0, 1]])


def transform_detections_sequence_to_car(x_seq: np.ndarray, y_seq: np.ndarray, odometry: np.ndarray):
    """
    Computes the transformation matrix from sequence coordinates (global coordinate system) to car coordinates.
    The position of the car is extracted from the odometry array.
    :param x_seq: Shape (n_detections,). Contains the x-coordinate of the detections in the sequence coord. system.
    :param y_seq: Shape (n_detections,). Contains the y-coordinate of the detections in the sequence coord. system.
    :param odometry: Numpy array containing at least the names fields "x_seq", "y_seq" and "yaw_seq" which give the
    position and orientation of the sensor vehicle.
    :return: Two 1D numpy arrays, both of shape (n_detections,). The first array contains the x-coordinate and the
    second array contains the y-coordinate of the detections in car coordinates.
    """
    trafo_matrix = trafo_matrix_seq_to_car(odometry)
    v = np.ones((len(x_seq), 3))
    v[:, 0] = x_seq
    v[:, 1] = y_seq
    res = batch_transform_3d_vector(trafo_matrix, v)
    return res[:, 0], res[:, 1]


def convert_to_anchor_coordinate(anchor_scene,scene):
    x_cc, y_cc = transform_detections_sequence_to_car(scene.radar_data["x_seq"], scene.radar_data["y_seq"],
                                                      anchor_scene.odometry_data)
    scene.sync_with_anchor(x_cc, y_cc)
    return scene


def gen_timeline(sequence):
    """
    a sequence_timeline is generated
    because the four radars has its own reference time, we need to sync all four of them
    this would require the anchor_timeline, which is the first radar in this sequence
    
    :param path: Sequence 
    
    :return: sequence_timeline, list; anchor_timeline, list
    """
    cur_sequence_timestamp = sequence.first_timestamp # initiate current sequence timestamp
    cur_anchor_timestamp = sequence.first_timestamp # initiate current anchor timestamp

    sequence_timeline = [cur_sequence_timestamp] #initiate sequence_timeline
    anchor_timeline = [cur_anchor_timestamp] #initiate anchor_timeline

    while True:
        cur_sequence_timestamp = sequence.next_timestamp_after(cur_sequence_timestamp) #sequentially read out all the sequence timestamps
        if cur_sequence_timestamp is None: # break at the end of the sequence
            break
        sequence_timeline.append(cur_sequence_timestamp) #append a sequence timestamp to sequence timeline

    
    while True:
        cur_anchor_timestamp = sequence.next_timestamp_after(cur_anchor_timestamp , same_sensor = True) #sequentially read out all the timestamps from the same radar
        if cur_anchor_timestamp is None: # break at the end of the sequence  这就是为什么9809不是除以4得到总帧数，而是大于这个数
            break
        anchor_timeline.append(cur_anchor_timestamp) #append an anchor timestamp to anchor timeline

    return sequence_timeline, anchor_timeline


def features_from_radar_data(radar_data):
    """
    generate a feature vector for each detection in radar_data.
    The spatial coordinates as well as the ego-motion compensated Doppler velocity and the RCS value are used.
    
    :param radar_data: input data    
    :return: numpy array with shape (len(radar_data), 5/4), contains the feature vector for each point
    """
    X = np.zeros((len(radar_data), 5))  # construct feature vector
    for radar_point_index in range(len(radar_data)):
        X[radar_point_index][0] = radar_data[radar_point_index]["x_cc"] #in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
        X[radar_point_index][1] = radar_data[radar_point_index]["y_cc"] #in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
        X[radar_point_index][2] = radar_data[radar_point_index]["vr_compensated"] #in m/s: Radial velocity for this detection but compensated for the ego-motion
        X[radar_point_index][3] = radar_data[radar_point_index]["rcs"] #in dBsm, Radar Cross Section value of the detection
        X[radar_point_index][4] = int(radar_data[radar_point_index]["track_id"], 16)
        # X[radar_point_index][4] = int(radar_data[radar_point_index]["uuid"], 16)
    return X
    # return X[:,0:4] #only take elemenet 1 to 3, noted that the index 4 means until but not included, this can be a point of confusion


def radar_scenes_dataset(datasetdir: str):
    """
    Given the path to a sequences.json file, all sequences from the category "training" are retrieved.
    
    :param datasetdir: path to the sequences.json file.
    
    :return: sequences
    """
    sequence_file_add = str(datasetdir)+"/sequences.json" # the address to the json file
    if not os.path.exists(sequence_file_add):
        print("Please make sure you entered the correct directory for sequences.json file.")

    with open(sequence_file_add, "r") as f: # read out the json file
        meta_data = json.load(f)

    sequ = []  # initialize sequence as a list
    for sequence_name, sequence_data in meta_data["sequences"].items():
        sequ.append(sequence_name)

    return sequ


def get_valid_points(scene, non_static = False):
    """
    Given a scene, filter out the clutters and outliers, only take into account the valid points
    
    :param scene: the particular scene where we want to perform the filtering with
    
    :return: valid point X and Y
    """
    radar_data = scene.radar_data 
    def clip(snippet:np.ndarray):
        '''
        remove points that not in -50<y_cc<50, 0<x_cc<100m
        '''
        idx = np.where(snippet["y_cc"]>=-50)[0] 
        clip_snip = snippet[idx]
        idx = np.where(clip_snip["y_cc"]<=50)[0]
        clip_snip = clip_snip[idx]
        idx = np.where(clip_snip["x_cc"]>=0)[0]
        clip_snip = clip_snip[idx] 
        idx = np.where(clip_snip["x_cc"]<=100)[0]
        clip_snip = clip_snip[idx]
        return clip_snip
    radar_data = clip(radar_data)
    y_true = np.array([ClassificationLabel.label_to_clabel(point) for point in radar_data["label_id"]]) #get all the labels
    id_true = np.array(radar_data["track_id"])
    valid_points = (y_true != None) & (id_true != b'')
    y_true = y_true[valid_points]  # filter out the invalid points
    y_true = np.array([point.value for point in y_true])
    id_true = id_true[valid_points]
    radar_data = radar_data[valid_points]

    if non_static:
        non_static_points = (y_true != 5)
        non_static_y_true = y_true[non_static_points]  # only keep the labels for valid points
        non_static_id_true = id_true[non_static_points]
        X = features_from_radar_data(radar_data[non_static_points]) #get the features from radar_data
        Y = np.row_stack((non_static_y_true, non_static_id_true))   # it contains track_id already！！！！！！
    else:
        X = features_from_radar_data(radar_data)  # get the features from radar_data
        Y = np.row_stack((y_true, id_true))

    return X, Y


# synchronize data collected by four radars into the ego_coordinate as seen by the anchor radar
def synchronize_global_coordinate_to_anchor_coordinate(frame_index: int, sequence: object, data: dict, label: dict, non_static=False):
    """
    Given a sequence, syncronize the four radars to generate sets of unifid points for this sequence 
    
    :param frame_index: this is the key for the dictionary, counting continuously from sequence 1 to 158
    :param sequence: the name of the sequence we perform this syncronization with
    :param data: a dictionary to put data in
    :para label: a dictionary to put label in
    :para non_static: a flag to indicate if we want to filter out the static points, such as trees and roads 
    
    :return: frame_index, so that it can be passon and count the frames countinously from sequence to sequence
    """

    sequence_timeline, anchor_timeline = gen_timeline(sequence) #first, generate timelines based on the training_sequence   
    anchor_point_count = 0
    for anchor_point in tqdm(anchor_timeline):  #synchronize all four radars based on the anchor_point  # 这里将四帧合成一帧！
        anchor_scene = sequence.get_scene(anchor_point) #get anchor_scene
        X, Y = get_valid_points(anchor_scene, non_static)
        point_count = sequence_timeline.index(anchor_point) #represent the index of anchor point in the regular time sequence, rather than in the anchor sequence. 
        for other_radar_index in range(3): #iterate the remaining 3 radars and synchronize each to that of the anchor
            if point_count+1+other_radar_index < len(sequence_timeline):
                cur_timestamp = sequence_timeline[point_count+ 1+other_radar_index] #from anchor_point+index+1 to get tha radar number            
                other_radar_scene = sequence.get_scene(cur_timestamp)  #get the scene from this radar
                synchronized_scene = convert_to_anchor_coordinate(anchor_scene, other_radar_scene) #synchronize, by converting the global coordinate of radar points to that of the ego_coordinate, as speficied by the anchor radar
                other_radar_X, other_radar_Y = get_valid_points(synchronized_scene, non_static)

            X = np.concatenate((X, other_radar_X),axis=0) #concatenate radar points to anchor radar points
            Y = np.concatenate((Y, other_radar_Y),axis=1) #concatenate labels

        data[frame_index] = X #register the data with frame_index
        label[frame_index] = Y #registre the label with frame_index
        frame_index += 1 #increase frame_index
        anchor_point_count+=1 #increase anchor point
    return frame_index


def radar_scenes_partitioned_data_generator(path_to_dataset: str, non_static = False):
    """
    partition the datasets into training data, validation data and testing data

    :param path_to_dataset: path to the dataset 
    :param non_static: indicate if we want to filter out the static points
    
    :return: the generated values
    """
    sequences_list = radar_scenes_dataset(path_to_dataset)
    print('Generate Data')
    data = {}  # initialize the data dictionary
    label = {}  # initialize the label dictionary
    index_prior = 0 #initialize frame_index
    for sequence_name in tqdm(sequences_list):
        try:
            sequ = Sequence.from_json(os.path.join(path_to_dataset, sequence_name, "scenes.json"))
        except FileNotFoundError:
            # if can't find the file path, prompt the following error message
            print('Please verify your path_to_dataset parameter')
        print('Processing {} for Data'.format(sequence_name))

        if non_static:
            index_post = synchronize_global_coordinate_to_anchor_coordinate(index_prior, sequ, data, label, non_static=True)
        else:
            index_post = synchronize_global_coordinate_to_anchor_coordinate(index_prior, sequ, data, label)

        index_prior =  index_post #update frame_index

    train_number = int(len(data) * 0) #.8
    validation_number = int(len(data) * 0) #.1
    keys = list(range(len(data)))
    train_data = {}
    train_label = {}
    validation_data = {}
    validation_label = {}
    test_data = {}
    test_label = {}
    for idx, key in enumerate(keys):
        if idx < train_number:  # keys的前train_number个key对应的元素放入train_dataset
            idx_train = idx
            train_data[idx_train] = data[key]
            train_label[idx_train] = label[key]
        elif idx < train_number + validation_number:  # keys接下来的validation_number个key对应的元素放入validation_dataset
            idx_validation = idx - train_number
            validation_data[idx_validation] = data[key]
            validation_label[idx_validation] = label[key]
        else:  # keys剩下的key对应的元素放入test_dataset
            idx_test = idx - train_number - validation_number
            test_data[idx_test] = data[key]
            test_label[idx_test] = label[key]

    assert list(train_data.keys()) == list(range(len(train_data)))
    assert list(validation_data.keys()) == list(range(len(validation_data)))
    assert list(test_data.keys()) == list(range(len(test_data)))

    # print out the partition of the dataset
    print("{} frames for training, {} frames for validation and {} frames for testing.".format(len(train_data),
                                                                                               len(validation_data),
                                                                                               len(test_data)))
    print("-" * 120)

    #store the generated data in pickle file
    if non_static:
        path = str(path_to_dataset) +'/train_data_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(train_data, f)
        f.close()

        path = str(path_to_dataset) +'/train_label_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(train_label, f)
        f.close()

        path = str(path_to_dataset) +'/validation_data_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(validation_data, f)
        f.close()

        path = str(path_to_dataset) +'/validation_label_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(validation_label, f)
        f.close()

        path = str(path_to_dataset) + '/test_data_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(test_data, f)
        f.close()

        path = str(path_to_dataset) + '/test_label_without_static.pickle'
        f = open(path, 'wb')
        pickle.dump(test_label, f)
        f.close()
    else:
        path = str(path_to_dataset) +'/train_data.pickle'
        f = open(path, 'wb')
        pickle.dump(train_data, f)
        f.close()

        path = str(path_to_dataset) +'/train_label.pickle'
        f = open(path, 'wb')
        pickle.dump(train_label, f)
        f.close()

        path = str(path_to_dataset) +'/validation_data.pickle'
        f = open(path, 'wb')
        pickle.dump(validation_data, f)
        f.close()

        path = str(path_to_dataset) +'/validation_label.pickle'
        f = open(path, 'wb')
        pickle.dump(validation_label, f)
        f.close()

        path = str(path_to_dataset) + '/test_data.pickle'
        f = open(path, 'wb')
        pickle.dump(test_data, f)
        f.close()

        path = str(path_to_dataset) + '/test_label.pickle'
        f = open(path, 'wb')
        pickle.dump(test_label, f)
        f.close()

    return train_data, validation_data, test_data, train_label, validation_label, test_label


def get_validation_data(path_to_dir, non_static=False):
    
    """
    check if there are data stored in the path, if not generate data

    :param path_to_dataset: path to the dataset 
    :param non_static: indicate if we want to filter out the static points
    
    :return: loaded data
    """

    if non_static:
        validation_data_add = str(path_to_dir) + "/validation_data_without_static.pickle"
        validation_label_add = str(path_to_dir) + "/validation_label_without_static.pickle"
    else:
        validation_data_add = str(path_to_dir) + "/validation_data.pickle"
        validation_label_add = str(path_to_dir) + "/validation_label.pickle"

    if not os.path.exists(validation_data_add):
        print("data directory is empty, calling data generator instead")
        _, validation_dataset, _, _, validation_label, _ = radar_scenes_partitioned_data_generator(path_to_dir, non_static)


    f_validation_data = open(validation_data_add, 'rb')
    validation_dataset=pickle.load(f_validation_data)
    f_validation_data.close()

    f_validation_label = open(validation_label_add, 'rb')
    validation_label =pickle.load(f_validation_label)
    f_validation_label.close()

    return validation_dataset ,  validation_label


def get_test_data(path_to_dir, non_static=False):
    """
    check if there are data stored in the path, if not generate data
    :param:
        path_to_dataset: path to the dataset
        non_static: indicate if we want to filter out the static points
    :return:
        loaded data
    """
    if non_static:
        test_data_add = str(path_to_dir) + "/test_data_without_static.pickle"
        test_label_add = str(path_to_dir) + "/test_label_without_static.pickle"
    else:
        test_data_add = str(path_to_dir) + "/test_data.pickle"
        test_label_add = str(path_to_dir) + "/test_label.pickle"

    if not os.path.exists(test_data_add):
        print("data directory is empty, calling data generator instead")
        _, _, test_dataset,_, _, test_label = radar_scenes_partitioned_data_generator(path_to_dir, non_static)
      
    f_test_data = open(test_data_add, 'rb')
    test_dataset = pickle.load(f_test_data)
    f_test_data.close()

    f_test_label = open(test_label_add, 'rb')
    test_label = pickle.load(f_test_label)
    f_test_label.close()

    return test_dataset, test_label


def hex_dex(trackid):
    return int(trackid, 16)
    



class Radar_Scenes_Validation_Dataset(Dataset):
    def __init__(self, datapath, transforms, sample_size, non_static):
        '''
        Define a class in order to interface with DataLoader

        Arguments
            - datapath: path to the Radar Scenes Dataset
            - transformes, as defined by the FeatureTransform file
            - sample_size, how many samples to take out from each scene
            - non_static: do we want to filter out the non_static points
        '''

        #load data
        validation_dataset, validation_label = get_validation_data(datapath, non_static)

        self.validation_dataset = validation_dataset
        self.validation_label = validation_label
        self.transforms =  transforms
        self.sample_size = sample_size


    def __getitem__(self,frame_index):
       
        # get the original data
        points = self.validation_dataset[frame_index] # read out the points contained  in this frame
        labels = self.validation_label[frame_index] # read out the labels contained in this frame
            
        num_points = len(points) # how many points are contained in this frame
        point_idxs = range(len(points)) # generate the index

        # sample a fixed length points from each frame, if sample_size == 0, use original points
        if self.sample_size == 0:
            selected_point_idxs = point_idxs
        elif self.sample_size >= num_points:
            selected_point_idxs = np.random.choice(point_idxs, self.sample_size, replace = True)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.sample_size, replace = False)
            
        selected_points = points[selected_point_idxs, :] #only take the sampled points in order to keep things of uniform size
        selected_labels = labels[:, selected_point_idxs]
        selected_labels = label_bytes2int(selected_labels)

        #transform the selected points, augmentation
        if self.transforms != None: 
            selected_points = self.transforms(selected_points)

        features = torch.tensor(np.stack(selected_points)).type(torch.FloatTensor)
        label   = torch.tensor(np.stack(selected_labels)).type(torch.FloatTensor)

        return features, label

    def __len__(self):
        return len(self.validation_dataset)


class Radar_Scenes_Test_Dataset(Dataset):
    def __init__(self, datapath, transforms, sample_size, non_static):
        '''
        Define a class in order to interface with DataLoader
        :param:
            - datapath: path to the Radar Scenes Dataset
            - transformes, as defined by the FeatureTransform file
            - sample_size, how many samples to take out from each scene. If sample_size == 0, use the original points
            - non_static: do we want to filter out the non_static points
        '''

        # load data
        test_dataset, test_label = get_test_data(datapath, non_static)
        
        self.test_dataset = test_dataset
        self.test_label = test_label
        self.transforms = transforms
        self.sample_size = sample_size

    def __getitem__(self, frame_index):
        # get the original data
        points = self.test_dataset[frame_index]  # read out the points contained  in this frame
        labels = self.test_label[frame_index]  # read out the labels contained in this frame
        labels[1, :] =  np.array(list(map(hex_dex, labels[1, :])))
        labels = labels.astype(float)

        num_points = len(points)  # how many points are contained in this frame
        if num_points == 0:
            return torch.empty(1), torch.empty(1)
        point_idxs = range(num_points)  # generate the index

        # 如果sample size == 0，不采样而直接使用原始点
        if self.sample_size == 0:
            selected_point_idxs = point_idxs
        # sample a fixed length points from each frame
        elif self.sample_size >= num_points:
            selected_point_idxs = np.random.choice(point_idxs, self.sample_size, replace=True)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.sample_size, replace=False)

        selected_points = points[selected_point_idxs, :]
        # only take the sampled points in order to keep things of uniform size
        selected_labels = labels[:, selected_point_idxs]

        # transform the selected points, augmentation
        if self.transforms is not None:
            selected_points = self.transforms(selected_points)

        features = torch.tensor(np.stack(selected_points)).type(torch.FloatTensor)
        label = torch.tensor(np.stack(selected_labels)).type(torch.FloatTensor)
        return features, label

    def __len__(self):
        return len(self.test_dataset)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Dataset generator')
    parser.add_argument('--datapath', help='File path of RadarScenes', type=str, default='../../RadarScenes/datashort')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    '''
    The main function
     1.generates .pickle fils if they do not exist;
     2.count how many empty frames in the validation set and test set
    '''
    args = parse_args()
    
    # radar_scenes_dataset = Radar_Scenes_Validation_Dataset(args.datapath, transforms=None, sample_size=100, non_static=True)
    # validationDataLoader = DataLoader(radar_scenes_dataset, batch_size=1, shuffle=False, num_workers=4)
    # num_empty_frames = 0
    # for idx, (features, label) in enumerate(validationDataLoader):
    #     if features.numel() == 1:
    #         print('frame id', idx, 'is empty')
    #         num_empty_frames += 1
    # print("{}%% frames in the validation set is empty".format(100*num_empty_frames/idx))
    
    radar_scenes_testset = Radar_Scenes_Test_Dataset(args.datapath, transforms=None, sample_size=100, non_static=True)
    testDataLoader = DataLoader(radar_scenes_testset, batch_size=1, shuffle=False, num_workers=4)
    num_empty_frames = 0
    for idx, (features, label) in enumerate(testDataLoader):
        if features.numel() == 1:
            print('frame id', idx, 'is empty')
            num_empty_frames += 1
    print("{}%% frames in the test set is empty".format(100*num_empty_frames/idx))
