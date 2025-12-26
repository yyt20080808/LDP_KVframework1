# coding=utf-8
import pandas as pd
from protocols.ours_two_d import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/
DATA_DIR = REPO_ROOT / "datasets"

def region_id_from_lonlat(lon, lat, lon_range, lat_range, n_lon, n_lat):
    """
    把 (lon, lat) 映射到 0..(n_lon*n_lat-1) 的区域编号。
    超出范围返回 -1。
    编号规则：region_id = lat_bin * n_lon + lon_bin （row-major）
    """
    lon0, lon1 = lon_range
    lat0, lat1 = lat_range

    # 在框内
    inside = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    rid = np.full(lon.shape[0], -1, dtype=np.int32)
    if not np.any(inside):
        return rid

    # 归一化到 [0,1]
    lon_norm = (lon[inside] - lon0) / (lon1 - lon0)
    lat_norm = (lat[inside] - lat0) / (lat1 - lat0)

    # 映射到 bin
    lon_bin = np.floor(lon_norm * n_lon).astype(np.int32)
    lat_bin = np.floor(lat_norm * n_lat).astype(np.int32)

    # 边界点落最后一个 bin
    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    lat_bin = np.clip(lat_bin, 0, n_lat - 1)

    rid[inside] = lat_bin * n_lon + lon_bin
    return rid


def heatmap_55x55_from_df(
        df: pd.DataFrame,
        lon_range=(-74.3, -73.4),
        lat_range=(40.5, 41.0),
        n_lon=11,
        n_lat=5,
        # as_matrix=True,   # True -> np.matrix; False -> np.ndarray
        log_scale=False,  # True -> 返回 log1p 后的浮点矩阵
):
    assert n_lon * n_lat == 55, "make sure n_lon * n_lat == 55"

    # 转数值 & 清洗
    dur = pd.to_numeric(df.get("trip_duration", pd.Series([0] * len(df))), errors="coerce")
    plon = pd.to_numeric(df["pickup_longitude"], errors="coerce")
    plat = pd.to_numeric(df["pickup_latitude"], errors="coerce")
    dlon = pd.to_numeric(df["dropoff_longitude"], errors="coerce")
    dlat = pd.to_numeric(df["dropoff_latitude"], errors="coerce")

    m = plon.notna() & plat.notna() & dlon.notna() & dlat.notna()
    m &= dur.isna() | (dur >= 0)

    if not m.any():
        H = np.zeros((55, 55), dtype=float if log_scale else np.int64)
        return H, [], []

    plonv = plon[m].to_numpy()
    platv = plat[m].to_numpy()
    dlonv = dlon[m].to_numpy()
    dlatv = dlat[m].to_numpy()

    pid = region_id_from_lonlat(plonv, platv, lon_range, lat_range, n_lon, n_lat)
    did = region_id_from_lonlat(dlonv, dlatv, lon_range, lat_range, n_lon, n_lat)

    ok = (pid >= 0) & (did >= 0)
    if not np.any(ok):
        H = np.zeros((55, 55), dtype=float if log_scale else np.int64)
        return H, [], []

    pid_ok = pid[ok].astype(np.int32)
    did_ok = did[ok].astype(np.int32)

    # 55×55 计数
    H = np.zeros((55, 55), dtype=np.int64)
    np.add.at(H, (pid_ok, did_ok), 1)

    if log_scale:
        H = np.log1p(H.astype(float))

    pickup_ids = pid_ok
    dropoff_ids = did_ok
    return H, pickup_ids, dropoff_ids


number_of_records = 1_458_196


def normalize_to_neg_one_to_one(a):
    min_val = 0
    max_val = 54

    #  [0, 1]
    normalized = (a - min_val) / (max_val - min_val)

    return normalized


def generate_data(trip_length):
    filepath = DATA_DIR / "taxi"/ "train_by_duration"/ f"train_dur_{trip_length}.csv"
    df_sss = pd.read_csv(filepath)
    heatmap, X, Y = heatmap_55x55_from_df(df_sss)
    # theta = heatmap.flatten()
    X = normalize_to_neg_one_to_one(X)
    Y = normalize_to_neg_one_to_one(Y)
    # plt.imshow(heatmap, cmap=plt.cm.bone, vmax=5000, vmin=00, interpolation='nearest', origin='lower',
    #                extent=[0, 54, 0, 54])
    #
    # plt.grid(True)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.show()
    return Y, X, heatmap


def OUE(data, epsilon):
    nall = sum(data)
    ee = np.exp(epsilon)
    res_cout = []
    p = 1 / 2
    q = 1 / (ee + 1)
    for tup in range(len(data)):
        v = data[tup]
        obs = np.random.binomial(v, p) + np.random.binomial(nall - v, q)
        res_cout.append(obs)
    recover_res = []
    # print(res_cout)
    for tup in range(len(data)):
        v = res_cout[tup]
        ll = (v - nall * q) / (p - q)
        # if ll < 0:
        #     recover_res.append(0)
        # else:
        recover_res.append(int(ll + 0.5))
    ns = np.array(recover_res)

    ns = ns.reshape((55, 55))
    return ns


from scipy.stats import pearsonr


def upper_triangle(mat):
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


if __name__ == "__main__":
    keys = ["0_300", "300_600", "600_900", "900_1200", "1200_1500", "1500_inf"]
    length = 55
    width = 55
    # epsilon_list = [1]
    epsilon_list = [0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]

    experi = 5 # time of running exeperiments
    m, n = len(keys), len(epsilon_list)
    sim_each_key_ours, sim_each_key_baselines = np.zeros((m, n)), np.zeros((m, n))
    fre_each_key_ours, fre_each_key_baselines = np.zeros((m, n)), np.zeros((m, n))
    i = 0
    for trip_key in keys:
        data1, data2, ori_data = generate_data(trip_key)
        real_f = len(data2) / number_of_records
        print(f"duration is:{trip_key} \t f is {round(real_f, 5)}")
        c = 0
        for ep in epsilon_list:
            sim_ours, frequency_ours = 0, 0
            sim_oue, frequency_oue = 0, 0
            for j in range(experi):
                res_ours, est_f_ours = twodimensionalSW(length, width, data1, data2, ep, number_of_records - len(data1))
                vA = upper_triangle(res_ours)
                vB = upper_triangle(ori_data)
                sim, _ = pearsonr(vA, vB)
                print(f"matrix correlation of ours = {sim}, est_f_mse = {(est_f_ours - real_f) ** 2}")
                sim_ours += sim
                frequency_ours += (est_f_ours - real_f) ** 2

                res_oue = OUE(ori_data.flatten(), ep)
                est_f_oue = res_oue.sum() / number_of_records
                vC = upper_triangle(res_oue)
                sim, _ = pearsonr(vC, vB)
                print(f"matrix correlation of OUE = {sim}, est_f_mse = {(est_f_oue - real_f) ** 2}")
                sim_oue += sim
                frequency_oue += (est_f_oue - real_f) ** 2
            sim_each_key_ours[i,c] = sim_ours/experi
            sim_each_key_baselines[i,c] = sim_oue/experi
            fre_each_key_ours[i,c] = frequency_ours/experi
            fre_each_key_baselines[i,c] = frequency_oue/experi
            c += 1
        i += 1


    # show results
    np.set_printoptions(formatter={'all': lambda x: str(x) + ', '})
    print("\n pearsonr similarity:")
    print("res_ours = ", np.mean(sim_each_key_ours, axis=0))
    print("res_baseline = ", np.mean(sim_each_key_baselines, axis=0))

    print("\nMSE of frequency:")
    print("res_ours = ", np.mean(fre_each_key_ours, axis=0))
    print("res_baseline = ", np.mean(fre_each_key_baselines, axis=0))