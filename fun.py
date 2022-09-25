from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import src.data.nuscenes.utils as nusc_utils
import pdb
import matplotlib.pyplot as plt

import os
from yacs.config import CfgNode
from shapely.strtree import STRtree
from collections import OrderedDict
from src.data.utils import get_visible_mask, get_occlusion_mask, transform, \
    encode_binary_labels
from nuscenes.eval.detection.constants import DETECTION_NAMES
import numpy as np
from pyquaternion import Quaternion



ROOT = os.path.abspath(os.path.join(__file__, '..'))


def load_config(config_path):
    with open(config_path) as f:
        return CfgNode.load_cfg(f)

def get_default_configuration():
    defaults_path = os.path.join(ROOT, 'configs/defaults.yml')
    return load_config(defaults_path)

def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in nusc_utils.STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)

        
        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)
    
    return map_data

def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    transform[:3, 3] = np.array(record['translation'])
    return transform

def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm) #! here error may exist # inv(pose) dot sensor

if __name__ == '__main__':


    # Load the default configuration
    config = get_default_configuration()
    config.merge_from_file('configs/datasets/nuscenes.yml')

    # Load NuScenes dataset
    dataroot = os.path.expandvars(config.dataroot)
    nuscenes = NuScenes(config.nuscenes_version, dataroot)

    # Preload NuScenes map data
    # map_data = { location : load_map_data(dataroot, location) 
    #              for location in nusc_utils.LOCATIONS }
    
    scene = nuscenes.scene[0]
    map_data = { location : load_map_data(dataroot, location) 
                 for location in nusc_utils.LOCATIONS }
    log = nuscenes.get('log', scene['log_token'])
    scene_map_data = map_data[log['location']]
    first_sample_token = scene['first_sample_token']
    sample_generator = nusc_utils.iterate_samples(nuscenes, first_sample_token)
    sample = sample_generator.__next__()
    
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)
    pdb.set_trace()
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)
    camera = nusc_utils.CAMERA_NAMES[0]
    sample_data = nuscenes.get('sample_data', sample['data'][camera])
    pdb.set_trace()

    
    
    nclass = len(DETECTION_NAMES) + 1
    pdb.set_trace()
    extents = config.map_extents
    resolution = config.map_resolution
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)
    pdb.set_trace()

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)
    pdb.set_trace()
    
    




    