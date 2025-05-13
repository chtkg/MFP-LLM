import torch
from metrics import All_Metrics
import json
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

def test(mode, mae_thresh=None, mape_thresh=0.0):
    len_nums = 0
    y_pred_in = []
    y_true_in = []
    y_pred_out = []
    y_true_out = []

    y_true_in_regionlist = []
    y_pred_in_regionlist = []
    y_true_out_regionlist = []
    y_pred_out_regionlist = []
    index_all = 0

    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")])

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_in = np.array(i_data["y_in"])
            y_out = np.array(i_data["y_out"])
            st_pre_infolow = np.array(i_data["st_pre_infolow"])
            st_pre_outfolow = np.array(i_data["st_pre_outfolow"])
            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all :
                len_nums = len_nums + 1
                y_true_in_region = np.stack(y_true_in, axis=-1)
                y_pred_in_region = np.stack(y_pred_in, axis=-1)
                y_true_out_region = np.stack(y_true_out, axis=-1)
                y_pred_out_region = np.stack(y_pred_out, axis=-1)
                y_true_in_regionlist.append(y_true_in_region)
                y_pred_in_regionlist.append(y_pred_in_region)
                y_true_out_regionlist.append(y_true_out_region)
                y_pred_out_regionlist.append(y_pred_out_region)
                y_pred_in = []
                y_true_in = []
                y_pred_out = []
                y_true_out = []
                index_all = i4data_all
            y_true_in.append(y_in)
            y_pred_in.append(st_pre_infolow)
            y_true_out.append(y_out)
            y_pred_out.append(st_pre_outfolow)
            if (i == len(data_t) - 1 and idx == len(file_list) - 1):
                y_true_in_region = np.stack(y_true_in, axis=-1)
                y_pred_in_region = np.stack(y_pred_in, axis=-1)
                y_true_out_region = np.stack(y_true_out, axis=-1)
                y_pred_out_region = np.stack(y_pred_out, axis=-1)
                y_true_in_regionlist.append(y_true_in_region)
                y_pred_in_regionlist.append(y_pred_in_region)
                y_true_out_regionlist.append(y_true_out_region)
                y_pred_out_regionlist.append(y_pred_out_region)
                y_pred_in = []
                y_true_in = []
                y_pred_out = []
                y_true_out = []
    print('len_nums', len_nums)
    y_true_in = np.stack(y_true_in_regionlist, axis=0)
    y_pred_in = np.stack(y_pred_in_regionlist, axis=0)
    y_true_out = np.stack(y_true_out_regionlist, axis=0)
    y_pred_out = np.stack(y_pred_out_regionlist, axis=0)
    y_pred_in, y_pred_out = np.abs(y_pred_in), np.abs(y_pred_out)
    print(y_true_in.shape, y_pred_in.shape, y_true_out.shape, y_pred_out.shape)

    if mode == 'regresion':
        mae_groups = []
        rmse_groups = []

        for group_start in range(0, y_true_in.shape[1], 3):
            mae_sum = 0
            rmse_sum = 0
            for offset in range(3):
                t = group_start + offset
                if t >= y_true_in.shape[1]:
                    continue
                mae_in, rmse_in, _, _, _ = All_Metrics(y_pred_in[:, t, ...], y_true_in[:, t, ...], mae_thresh, mape_thresh, None)
                mae_out, rmse_out, _, _, _ = All_Metrics(y_pred_out[:, t, ...], y_true_out[:, t, ...], mae_thresh, mape_thresh, None)
                mae_sum += mae_in + mae_out
                rmse_sum += (rmse_in + rmse_out) / 2

            mae_groups.append(mae_sum)
            rmse_groups.append(rmse_sum)

        for i, (mae_val, rmse_val) in enumerate(zip(mae_groups, rmse_groups)):
            print(f"Step{i + 1}, MAE={mae_val:.2f}, RMSE={rmse_val:.2f}")



################################ result path ################################
folder_path = ''

mode = 'regression'

test(mode)

