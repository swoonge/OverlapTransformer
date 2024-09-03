#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: generate the overlap-based ground truth file, such as the provided loop_gt_seq00_0.3overlap_inactive.npz.


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import time
from com_overlap_yaw import com_overlap_yaw
from utils import *
from tqdm import trange

# seqs = ["07_01", "07_02", "08_01", "08_02"]
seqs = ["08_01"]
data_folder = "/media/vision/Data0/DataSets/gm_datasets/"

for seq in seqs:
    print("processing seq: ", seq)
    # paths of kitti dataset
    # /media/vision/Data0/DataSets/kitti/dataset/sequences
    scan_folder = data_folder + seq + "/velodyne"
    calib_file = data_folder + seq + "/calib.txt"
    # prepare poses of semantic kitti dataset (refined poses)
    poses_file = data_folder + seq + "/poses.txt"

    scan_paths = load_files(scan_folder)
    poses = load_poses(poses_file)

    all_rows = []
    thresh = 0.3
    for i in trange(len(scan_paths)):
        # print(str(i) + "    -------------------------------->")
        time1 = time.time()
        scan_paths_this_frame = []
        poses_this_frame = []
        scan_paths_this_frame.append(scan_paths[i])
        poses_this_frame.append(poses[i])
        idx_in_range = []
        for idx in range(len(scan_paths)):
            if np.linalg.norm(poses[idx, :3, -1] - poses[i, :3, -1]) < 30 and (i-idx) > 100:
                scan_paths_this_frame.append(scan_paths[idx])
                poses_this_frame.append(poses[idx])
                idx_in_range.append(idx)
        # print("prepared indexes for current laser: ", idx_in_range)

        poses_new_this_frame = np.array(poses_this_frame)
        ground_truth_mapping = com_overlap_yaw(scan_paths_this_frame, poses_new_this_frame, frame_idx=0, leg_output_width=360)

        one_row = []
        for m in range(1, ground_truth_mapping.shape[0]):
            if ground_truth_mapping[m,2] > thresh:
                one_row.append(idx_in_range[m-1])
        all_rows.append(one_row)
        # print("gt list for current laser: ", one_row)
        time2 = time.time()
        # print("time: ", time2-time1)

    print(len(all_rows))
    all_rows_array = np.array(all_rows)
    np.savez_compressed(data_folder + seq + "/loop_gt_seq" + seq + "_0.3overlap_inactive", all_rows_array)


