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
parser.add_argument('--dataset_path', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', help='path to the scan data')


def plot_two_poses(map, current_pose_idx, matching_pose_idx, range_image1, range_image2, case="fp_case", vis=True):
    # 전체 지도 생성 -> x, y 좌표 추출
    x_map, y_map = zip(*[pose[:2, 3] for pose in map])

    # 현재 포즈 생성 -> x, y 좌표 추출
    x_current, y_current = x_map[current_pose_idx], y_map[current_pose_idx]

    # 매칭된 포즈 생성 -> x, y 좌표 추출
    x_matching, y_matching = x_map[matching_pose_idx], y_map[matching_pose_idx]

    # 전체 화면 창 설정
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(4, 1, figure=fig)

    # 1. 전체 지도와 포즈 플롯
    ax_map = fig.add_subplot(gs[0:2, :])  # 첫 번째 열 전체를 사용
    
    ax_map.plot(x_map[current_pose_idx:], y_map[current_pose_idx:], color='gray', linestyle='-', marker='.', label='Map', alpha=0.5)
    ax_map.plot(x_map[:current_pose_idx], y_map[:current_pose_idx], 'y.-', label='Map')  # 빨간색 점과 선
    # 2. Range Image 1
    ax_range1 = fig.add_subplot(gs[2, :])  # 두 번째 열의 첫 번째 행
    ax_range1.imshow(range_image1, cmap='gray')
    ax_range1.set_title('current_pose')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.legend()

    if case == "fp_case":
        ax_map.plot(x_current, y_current, 'bo', label='Current Pose')  # 파란색 점
        ax_map.plot(x_matching, y_matching, 'ro', label='Matched Pose') # 빨간색 점
        ax_map.set_title(case + " : " + str(current_pose_idx) + " - " + str(matching_pose_idx))
        # 3. Range Image 2
        ax_range2 = fig.add_subplot(gs[3, :])  # 두 번째 열의 두 번째 행
        ax_range2.imshow(range_image2, cmap='gray')
        ax_range2.set_title('matched_pose')
    elif case == "tp_case":
        ax_map.plot(x_current, y_current, 'bo', label='Current Pose')  # 파란색 점
        ax_map.plot(x_matching, y_matching, 'go', label='Matched Pose') # 초록색 점
        ax_map.set_title(case + " : " + str(current_pose_idx) + " - " + str(matching_pose_idx))
        # 3. Range Image 2
        ax_range2 = fig.add_subplot(gs[3, :])  # 두 번째 열의 두 번째 행
        ax_range2.imshow(range_image2, cmap='gray')
        ax_range2.set_title('matched_pose')
    elif case == "fn_case":
        ax_map.plot(x_current, y_current, color='coral', marker='.', label='Current Pose')  # 주황색 점
        ax_map.set_title(case + " : " + str(current_pose_idx))
    
    # 레이아웃 조정
    plt.tight_layout()
    # 이미지 저장
    plt.savefig(case + "/" + str(current_pose_idx) + "_" + str(matching_pose_idx) + '.png')

    # 플롯 표시
    if vis:
        plt.show()
    else:
        plt.close(fig)

def calculate_distance_matrix_fast(descriptors):
    # Calculate pairwise distances using pdist, then convert to a square form matrix
    distance_matrix = squareform(pdist(descriptors, 'euclidean'))
    return distance_matrix

def calculate_descriptor_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

def calculate_pose_distance(pose1, pose2):
    translation1 = pose1[:3, 3]
    translation2 = pose2[:3, 3]
    return np.linalg.norm(translation1 - translation2)

def calculate_pose_distances(poses):
    n = len(poses)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = calculate_pose_distance(poses[i], poses[j])
            distances[j, i] = distances[i, j]
    return distances

def find_matching_poses(poses, descriptors, descriptor_threshold, pose_threshold, exclusion_range=300):
    n = len(poses)
    matching_results = []
    
    # 미리 pose와 descriptor들 끼리의 거리 계산
    print("Calculating poses and descriptors distances ...")
    pose_distances = calculate_pose_distances(poses)
    descriptor_distances = calculate_distance_matrix_fast(descriptors)
    
    print("Finding matching poses ...")

    for i in tqdm(range(exclusion_range, n)):
        revisit = []
        match_candidates = []
        matches = []
        for j in range(0, i - exclusion_range):
            if descriptor_distances[i, j] < descriptor_threshold:
                match_candidates.append((j, pose_distances[i, j], descriptor_distances[i, j]))
            if pose_distances[i, j] < pose_threshold[0]:
                revisit.append(j)
        match_candidates.sort(key=lambda x: x[2])
        match_candidates = np.array(match_candidates)

        if match_candidates.shape[0] > 0: # Positive Prediction 
            for candidate in match_candidates:
                #  matching된 j     gt에 있는 j
                if candidate[1] <= pose_threshold[0]:
                    # True Positive (TP): 매칭에 성공하고, 그 pose가 실제 매칭되어야 하는 경우
                    matches.append((i, int(candidate[0]), candidate[1], candidate[2], "tp"))
                elif candidate[1] > pose_threshold[1]:
                    # False Positive (FP): 매칭에 성공했으나, 그 pose가 실제로 매칭되어야 하는 것이 아닌 경우
                    matches.append((i, int(candidate[0]), candidate[1], candidate[2], "fp"))
            if not matches: # 매칭된 모두가 3~20m 사이에 있어 tp, fp 모두 안된 경우. 이 경우는 거의 없다.
                if revisit:
                    # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                    matches.append((i, -1, -1, -1, "fn"))
                else:
                    # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                    matches.append((i, -1, -1, -1, "tn"))
        else: # Negative Prediction
            if revisit:
                # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                matches.append((i, -1, -1, -1, "fn"))
            else:
                # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                matches.append((i, -1, -1, -1, "tn"))
            

        matching_results.append(matches)
    
    return matching_results

def calculate_metrics(matching_results, top_k=5):
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    total_attempts = len(matching_results)
    
    topk_tp = 0  # Top-K Recall 계산을 위한 변수
    
    for matches in matching_results:
        first_match = matches[0]  # 첫 번째 매칭 결과
        
        if first_match[4] == "tp":
            tp += 1
        elif first_match[4] == "tn":
            tn += 1
        elif first_match[4] == "fp":
            fp += 1
        elif first_match[4] == "fn":
            fn += 1
        
        # Top-K Recall 계산 (상위 K개의 매칭에서 적어도 하나가 True Positive일 경우 성공으로 간주)
        if any(match[4] == "tp" for match in matches[:top_k]):
            topk_tp += 1
    
    # 메트릭 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_attempts if total_attempts > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    topk_recall = topk_tp / (topk_tp + fn) if (topk_tp + fn) > 0 else 0
    
    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1-Score": f1_score,
        "Accuracy": accuracy,
        "False Positive Rate (FPR)": fpr,
        "Top-{} Recall".format(top_k): topk_recall
    }

def __main__(sequence):
     # load scan paths and poses
    args = parser.parse_args()
    matrics_total = {}
    for seq in sequence:
        dataset_path = os.path.join(args.dataset_path, seq)

        # build model and load pretrained weights
        amodel = featureExtracter(channels=1, use_transformer=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amodel.to(device)
        print("Loading weights from ", test_weights)
        checkpoint = torch.load(test_weights)
        amodel.load_state_dict(checkpoint['state_dict'])
        amodel.eval()

        if not os.path.exists("preprocessed_data/range_images_" + seq + ".npy"):
            # load scans
            print("Loading scans ...")
            scan_paths = sorted(glob.glob(os.path.join(dataset_path, 'velodyne', '*.bin')))
            scans = []
            for scan_path in tqdm(scan_paths):
                scans.append(load_vertex(scan_path))
            
            # calculate range image
            print("Calculating range images ...")
            range_images = []
            for scan in tqdm(scans):
                proj_range, _, _, _ = range_projection(scan, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
                range_images.append(proj_range)
            range_images = np.array(range_images)
            np.save("preprocessed_data/range_images_" + seq + ".npy", range_images)
        else:
            print("Loading range_images ...")
            range_images = np.load("preprocessed_data/range_images_" + seq + ".npy")

        if not os.path.exists("preprocessed_data/descriptors_" + seq + ".npy"):
            # make discriptors for all range images
            print("Calculating descriptors ...")
            descriptors = []
            total_time = 0
            for range_image in tqdm(range_images):
                time_start = time.time()
                range_image_tensor = torch.from_numpy(range_image).unsqueeze(0)
                range_image_tensor = range_image_tensor.unsqueeze(0).cuda()
                descriptor = amodel(range_image_tensor).cpu().detach().numpy()
                time_end = time.time()
                total_time += time_end - time_start
                descriptors.append(np.squeeze(descriptor))
            mean_time = total_time / len(range_images)
            print(f"모델 평균 시간: {mean_time:.6f} 초")
            descriptors = np.array(descriptors)

            np.save("preprocessed_data/descriptors_" + seq + ".npy", descriptors)
        else:
            print("Loading descriptors ...")
            descriptors = np.load("preprocessed_data/descriptors_" + seq + ".npy")

        if not os.path.exists("preprocessed_data/poses_" + seq + ".npy"):
            # load calibrations
            calib_file = os.path.join(dataset_path, 'calib.txt')
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)

            # load poses
            poses_file = os.path.join(dataset_path, 'poses.txt')
            poses = load_poses(poses_file)
            pose0_inv = np.linalg.inv(poses[0])

            # for KITTI dataset, we need to convert the provided poses
            # from the camera coordinate system into the LiDAR coordinate system
            poses_new = []
            for pose in poses:
                poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
            # plot_two_poses(poses, poses_new)
            poses = np.array(poses_new)

            np.save("preprocessed_data/poses_" + seq + ".npy", poses)
        else:
            print("Loading poses ...")
            poses = np.load("preprocessed_data/poses_" + seq + ".npy")

        # descriptor_threshold = 0.3  # descriptor 유사성 임계값
        # descriptor_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # descriptor 유사성 임계값
        descriptor_thresholds = [0.3]  # descriptor 유사성 임계값
        pose_threshold = [3.0, 20.0]  # 실제 pose 거리 임계값 3m, 20m

        for distance_threshold in descriptor_thresholds:
            matching_results = find_matching_poses(poses, descriptors, distance_threshold, pose_threshold)
            metrics = calculate_metrics(matching_results, top_k=50)
            matrics_total[seq + "_" + str(distance_threshold)] = metrics

            # print("visualizing for FN") # 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
            # for matches in matching_results:
            #     first_match = matches[0]
            #     if first_match[4] == "fn":
            #         plot_two_poses(poses, first_match[0], first_match[1], range_images[first_match[0]], range_images[first_match[1]])

            print("visualizing for FP") # 매칭에 성공했지만, 실제로 매칭해야 하는 것이 없는 경우
            if not os.path.exists("fp_case"):
                os.makedirs("fp_case")
            if not os.path.exists("tp_case"):
                os.makedirs("tp_case")
            if not os.path.exists("fn_case"):
                os.makedirs("fn_case")
            for matches in matching_results:
                if not matches:
                    continue
                first_match = matches[0]
                if first_match[4] == "fp":
                    plot_two_poses(poses, first_match[0], first_match[1], range_images[first_match[0]], range_images[first_match[1]], "fp_case", False)
                if first_match[4] == "tp":
                    plot_two_poses(poses, first_match[0], first_match[1], range_images[first_match[0]], range_images[first_match[1]], "tp_case", False)
                if first_match[4] == "fn":
                    plot_two_poses(poses, first_match[0], first_match[1], range_images[first_match[0]], range_images[first_match[1]], "fn_case", False)

    print("[Total Metrics]")
    for key, value in matrics_total.items():
        print(f"Sequence {key}:")
        for key2, value2 in value.items():
            print(f"\t{key2}: {value2:.3f}")

if __name__ == '__main__':
    # use sequences 03–10 for training, sequence 02 for validation, and sequence 00 for evaluation.
    sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # sequence = ["00"]
    
    __main__(sequence)
