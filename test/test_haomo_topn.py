import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import numpy as np
import yaml


def cal_pr_curve(prediction_file_name, ground_truth_file_name, topn):

    des_dists = np.load(prediction_file_name)['arr_0']
    des_dists = np.asarray(des_dists, dtype='float32')
    des_dists = des_dists.reshape((len(des_dists), 3))

    ground_truth = np.load(ground_truth_file_name, allow_pickle='True')

    gt_num = 0
    all_num = 0
    check_out = 0
    for idx in range(0, 13500, 5):

        gt_idxes = np.array(ground_truth[int(gt_num)])
        gt_idxes =  gt_idxes[gt_idxes<12500]  # for forward

        if gt_idxes.any():
            all_num += 1
        else:
            gt_num += 1
            continue

        gt_num += 1

        dist_lists_cur = des_dists[des_dists[:,0]==idx,:]
        idx_sorted = np.argsort(dist_lists_cur[:,-1], axis=-1)
        for i in range(topn):
            if int(dist_lists_cur[idx_sorted[i], 1]) in gt_idxes:
                check_out += 1
                break
    print("top" + str(topn) + " recall: ", check_out / all_num)
    return check_out / all_num


def main(topn, ground_truth_file_name):

    prediction_file_name = "./test_results_haomo/predicted_des_L2_dis_bet_traj_forward.npz"

    topn_recall = cal_pr_curve(prediction_file_name, ground_truth_file_name, topn)

    return topn_recall


if __name__ == "__main__":
    # load config ================================================================
    config_filename = '../config/config_haomo.yml'
    config = yaml.safe_load(open(config_filename))
    ground_truth_file_name = config["file_root"]["gt_file"]
    # ============================================================================
    topn = 30

    recall_list = []
    for i in range(topn):
        print(i)
        rec = main(i, ground_truth_file_name)
        recall_list.append(rec)
    print(recall_list)
    np.save("./test_results_haomo/recall_list", np.array(recall_list))
