# coding=utf-8

import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/
DATA_DIR = REPO_ROOT / "datasets"

from protocols.ours import OURS_UE
from protocols.pckv_ue import pckv_UE
from protocols.pckv_grr import pckv_GRR
from protocols.hio_olh import hio

def read_padding_and_random_sampling(filepath):
    all_lists = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 由于每行是以 \t 开头写的，这里 split 后会有空字符串
            items = line.split('\t')
            # 过滤空字符串并转成 int
            row = [float(x) for x in items if x != ""]
            all_lists.append(row)
    return all_lists


def generate_Data(distribution_type):
    if distribution_type == "gaussian":
        path =  DATA_DIR / "gaussian_dataset" / "padding_and_sampling.txt"
        print("Information of Gaussian distribution:")
    else:
        path = DATA_DIR / "powerlaw_dataset" / "padding_and_sampling.txt"
        print("Information of Power-law distribution:")
    all_lists = read_padding_and_random_sampling(path)
    print(f"Number of keys reported(Including dummy):{len(all_lists)}")
    print(f"Real Reports on each key {[len(i) for i in all_lists]}")
    return all_lists


if __name__ == "__main__":
    # all_list = generate_Data( "gaussian")
    all_list = generate_Data("power-law")
    padding_l = 6
    number = 1000000
    epslions = [2]

    m = len(epslions)
    n = 10
    MEGRR_f, MEOLH_f, PCKVUE_f, JVEPM_f, ours_f = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros((n, m))
    MEGRR_v, MEOLH_v, PCKVUE_v, JVEPM_v, ours_v = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros((n, m))
    MEGRR_sum, MEOLH_sum, PCKVUE_sum, JVEPM_sum, ours_sum = np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros(
        (n, m)), np.zeros((n, m))

    for i in range(n):
        newa = all_list[i]
        newb = [0 for _ in range(number - len(newa))]
        data = [newa, newb]
        mse_MEGRR_f, mse_MEGRR_v, mse_MEGRR_sum = pckv_GRR(data, epslions, mul_d=100,paddinglength=padding_l)
        mse_MEOLH_f, mse_MEOLH_v, mse_MEOLH_sum = hio(data, epslions,paddinglength=padding_l)
        mse_PCKVUE_f, mse_PCKVUE_v, mse_PCKVUE_sum = pckv_UE(data, epslions,paddinglength=padding_l)
        mse_SVE_f, mse_SVE_EM_v, mse_SVE_EM_sum,res_range = OURS_UE(data, epslions,paddinglength=padding_l)
    #
        for j in range(len(epslions)):
            MEGRR_f[i, j] = round(mse_MEGRR_f[j], 10)
            MEGRR_v[i, j] = round(mse_MEGRR_v[j], 10)
            MEGRR_sum[i, j] = round(mse_MEGRR_sum[j], 10)
    #         # #
            MEOLH_f[i, j] = round(mse_MEOLH_f[j],10)
            MEOLH_v[i, j] = round(mse_MEOLH_v[j],10)
            MEOLH_sum[i,j] = round(mse_MEOLH_sum[j],10)
    #         # #
            PCKVUE_f[i, j] = round(mse_PCKVUE_f[j], 10)
            PCKVUE_v[i, j] = round(mse_PCKVUE_v[j], 10)
            PCKVUE_sum[i, j] = round(mse_PCKVUE_sum[j], 10)
    #         # # JVEPM_f[i, j] = mse_JVEPM_f[j]
    #         # # JVEPM_v[i, j] = mse_JVEPM_v[j]
    #         # # #
            ours_f[i, j] = round(mse_SVE_f[j], 10)
            ours_v[i, j] = round(mse_SVE_EM_v[j], 10)
            ours_sum[i, j] = round(mse_SVE_EM_sum[j], 10)
    #
    print("\nMSE of frequency:")
    print("res_ME_grr = ", np.mean(MEGRR_f, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_f, axis=0))
    print("PCKV_UE = ", np.mean(PCKVUE_f, axis=0))
    print("ours_f = ", np.mean(ours_f, axis=0))
    print("\nMSE of values mean:")
    print("res_ME_grr = ", np.mean(MEGRR_v, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_v, axis=0))
    print("res_PCKV_UE = ", np.mean(PCKVUE_v, axis=0))
    print("OURS_UE = ", np.mean(ours_v, axis=0))

    print("\nMSE of values sum:")
    print("res_ME_grr = ", np.mean(MEGRR_sum, axis=0))
    print("res_ME_OLH = ", np.mean(MEOLH_sum, axis=0))
    print("res_PCKV_UE = ", np.mean(PCKVUE_sum, axis=0))
    print("OURS_UE = ", np.mean(ours_sum, axis=0))
    # for i in range(n):
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
