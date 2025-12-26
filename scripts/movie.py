# coding=utf-8

import numpy as np
from protocols.ours import OURS_UE, getrangemse
# import matplotlib.pyplot as plt
from protocols.pckv_ue import pckv_UE
from protocols.pckv_grr import pckv_GRR
from protocols.hio_olh import hio
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/ 的上一级就是根目录
DATA_DIR = REPO_ROOT / "datasets"
# https://grouplens.org/datasets/movielens/

# 32 million records， 87,585 movies by 200,948 users
number_of_users = 200948


def read_padding_file(popular_IDs, ifours=True):
    if ifours:
        filename = DATA_DIR/"movie"/"movie_paddingSimple_Ours.txt"
        l = 10
    else:
        filename = DATA_DIR/"movie"/"movie_paddingSimple_100.txt"
        l = 100
    first_column_set = set(popular_IDs)
    data_dict = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            if int(col1) in first_column_set:
                data_dict[int(col1)].append(float(col2))
        file.close()
    for k, vals in data_dict.items():
        arr = np.array(vals, dtype=np.float32)
        data_dict[k] = normalize_to_neg_one_to_one(arr).tolist()
    return data_dict, l


def normalize_to_neg_one_to_one(a):
    min_val = 0
    max_val = 5

    # 处理所有元素相同的情况
    if max_val == min_val:
        return np.zeros_like(a)

    # 线性归一化到 [0, 1]
    normalized = (a - min_val) / (max_val - min_val)
    # 映射到 [-1, 1]
    normalized = 2 * normalized - 1
    return normalized


def read_true_statsics():
    frequency = {}
    mean = {}
    path = DATA_DIR/"movie"/"movie_top_f.txt"
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, cnt = map(int, line.split())
            frequency[idx] = cnt / number_of_users
        f.close()
    path = DATA_DIR/"movie"/"movie_top_mean.txt"
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, cnt = map(float, line.split())
            mean[idx] = cnt
        f.close()
    for k, vals in mean.items():
        arr = np.array(vals, dtype=np.float32)
        mean[k] = normalize_to_neg_one_to_one(arr).tolist()
    return frequency, mean


if __name__ == "__main__":
    k_pop_IDs = [1, 47, 50, 150, 216, 231, 256, 260, 296, 318, 349, 356, 480, 527, 593, 628, 661, 788, 858, 899, 919,
                 953, 1036, 1073, 1210, 1235, 1617, 1721, 1884, 1917, 1923, 1968, 2020, 2571, 2694, 2871, 2959, 2997,
                 3555, 4226, 4239, 4306, 4993, 5464, 5952, 7153, 8368, 8464, 30749, 48385, 53125, 56367, 57669, 79132,
                 79702, 106487]
    #
    # epslions = [2]
    epslions = [0.5,1, 1.5, 2, 2.5, 3, 4]
    m = len(epslions)
    muld = 1000
    our_data, padding_l_our = read_padding_file(k_pop_IDs, True)
    pckv_data, padding_l = read_padding_file(k_pop_IDs, False)
    real_f_lists, real_mean_lists = read_true_statsics()
    keys = real_f_lists.keys()
    n = 30
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
        print(f"key is :{idx}")
        data = pckv_data.get(idx)
        if data != None:
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
