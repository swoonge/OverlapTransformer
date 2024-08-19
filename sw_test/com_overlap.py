#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
import matplotlib.pyplot as plt
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *

def com_overlap(scan_paths, poses, frame_idx):
  # init ground truth overlap and yaw
  print('Start to compute ground truth overlap ...')
  overlaps = []

  # we calculate the ground truth for one given frame only
  # generate range projection for the given frame
  current_points = load_vertex(scan_paths[frame_idx])
  current_range, project_points, _, _ = range_projection(current_points, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
  visible_points = project_points[current_range > 0]
  valid_num = len(visible_points)
  current_pose = poses[frame_idx]

  tau_ = 1.2
  print("threshold for overlap: ", tau_)

  reference_range_list = []
  for i in range(len(scan_paths)):
    # generate range projection for the reference frame
    reference_idx = int(scan_paths[i][-10:-4])
    reference_pose = poses[reference_idx]
    reference_points = load_vertex(scan_paths[i])

    reference_points_world = reference_pose.dot(reference_points.T).T
    reference_points_in_current = np.linalg.inv(current_pose).dot(reference_points_world.T).T # 여기서 원점으로 pointcloud를 projection해야 하는데 안됨.
    
    reference_range, _, _, _ = range_projection(reference_points_in_current, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp_pc = reference_points_world[np.random.choice(len(reference_points_world), 10000, replace=False)]
    disp_pc2 = reference_points_in_current[np.random.choice(len(reference_points_world), 10000, replace=False)]
    disp_pc = reference_points_world
    disp_pc2 = reference_points_in_current
    ax.scatter(disp_pc[:, 0], disp_pc[:, 1], c='r')
    ax.scatter(disp_pc2[:, 0], disp_pc2[:, 1], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.show()

    # calculate overlap
    overlap = np.count_nonzero(abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < tau_) / valid_num
    overlaps.append(overlap)
    reference_range_list.append(reference_range)


  # ground truth format: each row contains [current_frame_idx, reference_frame_idx, overlap,]
  ground_truth_mapping = np.zeros((len(scan_paths), 3))
  ground_truth_mapping[:, 0] = np.ones(len(scan_paths)) * frame_idx
  ground_truth_mapping[:, 1] = np.arange(len(scan_paths))
  ground_truth_mapping[:, 2] = overlaps

  print('Finish generating ground_truth_mapping!')
  
  return ground_truth_mapping, current_range, reference_range_list
