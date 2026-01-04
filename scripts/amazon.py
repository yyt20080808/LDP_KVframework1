# coding=utf-8

import numpy as np
import random
from protocols.ours import OURS_UE, getrangemse
# import matplotlib.pyplot as plt
from protocols.pckv_ue import pckv_UE
from protocols.pckv_grr import pckv_GRR
from protocols.hio_olh import hio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/
DATA_DIR = REPO_ROOT / "datasets"
from collections import defaultdict


# 2 million records，240000 products by 1210271 users
number_of_users = 1210271


def read_padding_file(popular_IDs, ifours=True):
    if ifours:
        filename = DATA_DIR/"amazonshopping"/"amazon_paddingSimple_Ours.txt"
        l = 1
    else:
        filename = DATA_DIR/"amazonshopping"/"amazon_paddingSimple_3.txt"
        l = 3
    first_column_set = set(popular_IDs)
    data_dict = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            if col1 in first_column_set:
                data_dict[col1].append(float(col2))
        file.close()
    for k, vals in data_dict.items():
        arr = np.array(vals, dtype=np.float32)
        data_dict[k] = normalize_to_neg_one_to_one(arr).tolist()
    return data_dict, l


def normalize_to_neg_one_to_one(a):
    min_val = 0
    max_val = 5

    #  [0, 1]
    normalized = (a - min_val) / (max_val - min_val)
    # map to [-1, 1]
    normalized = 2 * normalized - 1
    return normalized


def read_true_statsics():
    frequency = {}
    mean = {}
    path = DATA_DIR/"amazonshopping"/"amazon_top_f.txt"
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, cnt = map(str, line.split())
            frequency[idx] = int(cnt) / number_of_users
        f.close()
    path = DATA_DIR / "amazonshopping" / "amazon_top_mean.txt"
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, cnt = map(str, line.split())
            mean[idx] = float(cnt)
        f.close()
    for k, vals in mean.items():
        arr = np.array(vals, dtype=np.float32)
        mean[k] = normalize_to_neg_one_to_one(arr).tolist()
    return frequency, mean

if __name__ == "__main__":
    k_pop_IDs = ['B00176B9JC', 'B000ASDGK8', 'B000UVZU1S', 'B004Z40048', 'B001AO0WCG', 'B0046VGPHQ', 'B001MA0QY2', 'B002WTC38O',
                 'B00147FGJ8', 'B0056GDG90', 'B004OHQR1Q', 'B007BLN17K', 'B001RMP7M6', 'B003BQ6QXK', 'B0047ETXD0',
                 'B00121UVU0', 'B0069FDR96', 'B000142FVW', 'B000FS05VG', 'B003V265QW', 'B003S516XO', 'B006SVCY6I',
                 'B000TKH6G2', 'B0002JKPA4', 'B0043OYFKU', 'B00150LT40', 'B002JSL6QI', 'B000L596FE', 'B008LDFU26',
                 'B000ZMBSPE', 'B0009FHJRS', 'B000VPPUEA', 'B00006IV2F', 'B00538TSMU', 'B00021DVCQ', 'B002LB75AO',
                 'B0000YUXI0', 'B0048O2R1E', 'B0009V1YR8', 'B008U12YV4', 'B000OYJ9AO', 'B004VFXVJW', 'B00639DLV2',
                 'B0001ZYLAO', 'B008MP481M', 'B009T47YZ2', 'B00IALDHDI', 'B005BF1M10', 'B001JKTTVQ', 'B003UH0528',
                 'B000UM2KCY', 'B000EVGQ0S', 'B006L1DNWY', 'B004GQZX4M', 'B004INUWX0', 'B00188IFHS', 'B00I073WLE',
                 'B00067YSLO', 'B0058E3XJI', 'B00CFRBIC0', 'B009GYVMAS', 'B006IBM21K', 'B002JPJ2ZS', 'B009RNUH4A',
                 'B007Q0WW0S', 'B001UHN0I6', 'B007TNHQOY', ]

    #
    # epslions = [0.5,1, 1.5, 2, 2.5, 3, 4]
    epslions = [2]
    m = len(epslions)
    muld = 10000
    our_data, padding_l_our = read_padding_file(k_pop_IDs, True)
    pckv_data, padding_l = read_padding_file(k_pop_IDs, False)
    real_f_lists, real_mean_lists = read_true_statsics()
    keys = real_f_lists.keys()
    n = 5
    MEGRR_f, MEOLH_f, PCKVUE_f, JVEPM_f, ours_f = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros((n, m))
    MEGRR_v, MEOLH_v, PCKVUE_v, JVEPM_v, ours_v = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros((n, m))
    MEGRR_sum, MEOLH_sum, PCKVUE_sum, JVEPM_sum, ours_sum = np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros(
        (n, m)), np.zeros((n, m))

    ours_rangeq = np.zeros((n, m))
    uniform_rangeq = 0
    i = 0
    for idx in real_f_lists.keys():
        print(f"key is {idx}")
        data = pckv_data.get(idx)
        real_f = real_f_lists[idx]
        real_mean = real_mean_lists[idx]
        data = [data, [0 for _ in range(number_of_users - len(data))]]
        mse_MEGRR_f, mse_MEGRR_v, mse_MEGRR_sum = pckv_GRR(data, epslions, mul_d=muld, real_f=real_f,
                                                           real_mean=real_mean, paddinglength=padding_l)
        mse_MEOLH_f, mse_MEOLH_v, mse_MEOLH_sum = hio(data, epslions, real_f=real_f, real_mean=real_mean,
                                                      paddinglength=padding_l)
        mse_PCKVUE_f, mse_PCKVUE_v, mse_PCKVUE_sum = pckv_UE(data, epslions, real_f=real_f, real_mean=real_mean,
                                                             paddinglength=padding_l)
        data2 = our_data[idx]
        data2 = [data2, [0 for _ in range(number_of_users - len(data2))]]
        # print(len(data2[0]), len(data2[1]))
        mse_SVE_f, mse_SVE_EM_v, mse_SVE_EM_sum, mse_rangeq = OURS_UE(data2, epslions, real_f=real_f,
                                                                      real_mean=real_mean, paddinglength=padding_l_our,
                                                                      smooth=False)

        for j in range(len(epslions)):
            MEGRR_f[i, j] = round(mse_MEGRR_f[j], 10)
            MEGRR_v[i, j] = round(mse_MEGRR_v[j], 10)
            MEGRR_sum[i, j] = round(mse_MEGRR_sum[j], 10)
            # #
            MEOLH_f[i, j] = round(mse_MEOLH_f[j], 10)
            MEOLH_v[i, j] = round(mse_MEOLH_v[j], 10)
            MEOLH_sum[i, j] = round(mse_MEOLH_sum[j], 10)
            # # # # #
            PCKVUE_f[i, j] = round(mse_PCKVUE_f[j], 10)
            PCKVUE_v[i, j] = round(mse_PCKVUE_v[j], 10)
            PCKVUE_sum[i, j] = round(mse_PCKVUE_sum[j], 10)

            # # # #
            ours_f[i, j] = round(mse_SVE_f[j], 10)
            ours_v[i, j] = round(mse_SVE_EM_v[j], 10)
            ours_sum[i, j] = round(mse_SVE_EM_sum[j], 10)
            ours_rangeq[i, j] = round(mse_rangeq[j], 10)
        uniform_rangeq = getrangemse(data[0])
        i += 1
        if i == n:
            break
    np.set_printoptions(formatter={'all': lambda x: str(x) + ', '})
    print("\nMSE of frequency:")
    print("res_ME_grr = ", np.mean(MEGRR_f, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_f, axis=0))
    print("res_PCKV_UE = ", np.mean(PCKVUE_f, axis=0))
    print("res_ours = ", np.mean(ours_f, axis=0))

    print("\nMSE of values mean:")
    print("res_ME_grr = ", np.mean(MEGRR_v, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_v, axis=0))
    print("res_PCKV_UE = ", np.mean(PCKVUE_v, axis=0))
    print("res_ours = ", np.mean(ours_v, axis=0))

    print("\nMSE of values sum:")
    print("res_ME_grr = ", np.mean(MEGRR_sum, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_sum, axis=0))
    print("res_PCKV_UE = ", np.mean(PCKVUE_sum, axis=0))
    print("res_ours = ", np.mean(ours_sum, axis=0))

    print("\nMSE of values range:")
    print("res_ours = ", np.mean(ours_rangeq, axis=0))
    print("uniform_rangeq = ", uniform_rangeq)
    # for i in range(len(portion)):
    #     print("frequency: portion", portion[i])
    #     print("res_ME_grr = ", MEGRR_f[i, :])
    #     print("res_ME_OLH = ", MEOLH_f[i, :])
    #     print("res_PCKV_UE = ", PCKVUE_f[i, :])
    #     print("ours_f = ", ours_f[i, :])
    #     print("\nMSE of values mean:")
    #     print("res_ME_grr = ", MEGRR_v[i, :])
    #     print("res_ME_OLH = ", MEOLH_v[i, :])
    #     print("res_PCKV_UE = ", PCKVUE_v[i, :])
    #     print("OURS_ = ", ours_v[i, :])
    #
    #     print("\nMSE of values sum:")
    #     print("res_ME_grr = ", MEGRR_sum[i, :])
    #     print("res_ME_OLH = ", MEOLH_sum[i, :])
    #     print("res_PCKV_UE = ", PCKVUE_sum[i, :])
    #     print("OURS_UE = ", ours_sum[i, :])
