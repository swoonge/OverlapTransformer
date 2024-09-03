# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
import glob
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import yaml
from com_overlap import com_overlap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy.spatial.distance import pdist, squareform

from modules.overlap_transformer import featureExtracter

# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
test_weights = config["demo1_config"]["test_weights"]
# ============================================================================

# set args for the script
parser = argparse.ArgumentParser(description='Generate overlap and orientation combined mapping file')
parser.add_argument('--dataset_path', type=str, default='/media/vision/Data0/DataSets/gm_datasets/', help='path to the scan data')

def plot_range_img(range_image):

    # 전체 화면 창 설정
    plt.figure(figsize=(18, 5))

    # 지도 플롯
    plt.imshow(range_image, cmap='gray')
    plt.title("Range image")
    plt.xlabel('H')
    plt.ylabel('W')
    plt.show()

def __main__(sequence):
     # load scan paths and poses
    args = parser.parse_args()
    matrics_total = {}
    for seq in sequence:
        dataset_path = os.path.join(args.dataset_path, seq)

        # load scans
        print("Loading scans ...")
        scan_paths = sorted(glob.glob(os.path.join(dataset_path, 'velodyne', '*.bin')))[1000:1002]
        print(scan_paths)
        scans = []
        for scan_path in tqdm(scan_paths):
            scans.append(load_vertex(scan_path))
            
        
        # calculate range image
        print("Calculating range images ...")
        fov_ups = np.arange(2.0, 20.0, 0.25)
        fov_downs = np.arange(-10.0, 0.0, 0.25)
        scan = scans[0]
        pixel_score = {}
        for fov_up in fov_ups:
            for fov_down in fov_downs:
                print("fov_up: ", fov_up, " || fov_down: ", fov_down)
                proj_range, _, _, _ = range_projection(scan, fov_up=fov_up, fov_down=fov_down, proj_H=32, proj_W=900, max_range=50)
                count = np.sum(proj_range > -1)
                pixel_score[(fov_up, fov_down)] = count
                # plot_range_img(proj_range)

        sorted_parameters = sorted(pixel_score, key=pixel_score.get, reverse=True)
        print(sorted_parameters[:10])
        # >> [(3.75, -7.75), (5.5, -5.75), (3.75, -8.0), (3.75, -7.5), (3.75, -7.0), (3.75, -6.75), (3.75, -6.5), (4.0, -7.0), (4.0, -6.75), (4.0, -6.5)]

        # plot pixel_score
        # x축과 y축 값을 추출
        x_vals = sorted(set([k[0] for k in pixel_score.keys()]))
        y_vals = sorted(set([k[1] for k in pixel_score.keys()]))
        z_vals = np.zeros((len(y_vals), len(x_vals)))

        # 각 (x, y) 좌표에 해당하는 값을 2D 배열에 채움
        for (x, y), score in pixel_score.items():
            x_index = x_vals.index(x)
            y_index = y_vals.index(y)
            z_vals[y_index, x_index] = score

        # 히트맵을 생성
        plt.figure(figsize=(10, 8))
        plt.imshow(z_vals, cmap='viridis', aspect='auto')

        # 축 레이블 설정
        plt.xticks(ticks=np.arange(len(x_vals)), labels=x_vals)
        plt.yticks(ticks=np.arange(len(y_vals)), labels=y_vals)

        # 컬러바 추가
        plt.colorbar(label='Score')

        # 제목 추가
        plt.title('Model Scores Heatmap')
        plt.xlabel('Parameter 1')
        plt.ylabel('Parameter 2')

        # 플롯 표시
        plt.show()

        
                


if __name__ == '__main__':
    # use sequences 03–10 for training, sequence 02 for validation, and sequence 00 for evaluation.
    # sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    sequence = ["08_01"]
    
    __main__(sequence)
