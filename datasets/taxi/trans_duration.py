import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def region_id_from_lonlat(lon, lat, lon_range, lat_range, n_lon, n_lat):
    """
    把 (lon, lat) 映射到 0..(n_lon*n_lat-1) 的区域编号。
    超出范围返回 -1。
    编号规则：region_id = lat_bin * n_lon + lon_bin （row-major）
    """
    lon0, lon1 = lon_range
    lat0, lat1 = lat_range

    # mask：在框内
    inside = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    rid = np.full(lon.shape[0], -1, dtype=np.int32)
    if not np.any(inside):
        return rid

    # 归一化到 [0,1]，再映射到 bin
    # 注意：lon==lon1 或 lat==lat1 的点，落到最后一个 bin
    lon_norm = (lon[inside] - lon0) / (lon1 - lon0)
    lat_norm = (lat[inside] - lat0) / (lat1 - lat0)

    lon_bin = np.floor(lon_norm * n_lon).astype(np.int32)
    lat_bin = np.floor(lat_norm * n_lat).astype(np.int32)

    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    lat_bin = np.clip(lat_bin, 0, n_lat - 1)

    rid[inside] = lat_bin * n_lon + lon_bin
    return rid


def pickup_dropoff_heatmap_55x55_by_trip_duration(
    train_path="train.csv",
    chunksize=100_000,
    lon_range=(-74.3, -73.4),
    lat_range=(40.5, 41.0),
    n_lon=11,
    n_lat=5,  # 11*5=55
    duration_bins=(0, 300, 600, 900, 1200, 1500, np.inf),  # [0,300),...[1500,inf)
    log_scale=True,
    vmin=0,
    vmax=10,  # log 后的 vmax（想统一色条就固定；否则可改成 None 自动）
):
    assert n_lon * n_lat == 55, "请确保 n_lon * n_lat == 55（比如 11×5）"

    usecols = [
        "trip_duration",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]

    # 时长段标签
    duration_bins = np.array(duration_bins, dtype=float)
    labels = [
        f"{int(duration_bins[i])}~{int(duration_bins[i+1])}"
        if np.isfinite(duration_bins[i+1]) else f"{int(duration_bins[i])}+"
        for i in range(len(duration_bins) - 1)
    ]

    # 每个时长段一张 55×55 矩阵
    M = {lab: np.zeros((55, 55), dtype=np.int64) for lab in labels}

    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunksize):
        dur = pd.to_numeric(chunk["trip_duration"], errors="coerce")
        plon = pd.to_numeric(chunk["pickup_longitude"], errors="coerce")
        plat = pd.to_numeric(chunk["pickup_latitude"], errors="coerce")
        dlon = pd.to_numeric(chunk["dropoff_longitude"], errors="coerce")
        dlat = pd.to_numeric(chunk["dropoff_latitude"], errors="coerce")

        m = dur.notna() & plon.notna() & plat.notna() & dlon.notna() & dlat.notna()
        m &= (dur >= 0)

        if not m.any():
            continue

        durv = dur[m].to_numpy()
        plonv = plon[m].to_numpy()
        platv = plat[m].to_numpy()
        dlonv = dlon[m].to_numpy()
        dlatv = dlat[m].to_numpy()

        # 映射到 0..54 的区域编号（超出为 -1）
        pid = region_id_from_lonlat(plonv, platv, lon_range, lat_range, n_lon, n_lat)
        did = region_id_from_lonlat(dlonv, dlatv, lon_range, lat_range, n_lon, n_lat)

        ok = (pid >= 0) & (did >= 0)
        if not np.any(ok):
            continue

        pid = pid[ok]
        did = did[ok]
        durv = durv[ok]

        # duration 分箱（left-closed right-open）
        cat = pd.cut(
            durv,
            bins=duration_bins,
            right=False,
            labels=labels,
            include_lowest=True
        )

        valid = pd.notna(cat)
        if not np.any(valid):
            continue

        pid = pid[valid]
        did = did[valid]
        cat = cat[valid]

        # 累计：对每个 label，把 (pid, did) 计数加 1
        # 用 np.add.at 做高效 scatter add
        for lab in pd.unique(cat):
            lab = str(lab)
            mm = (cat == lab)
            np.add.at(M[lab], (pid[mm], did[mm]), 1)

    # 绘图：每个时长段一张 55×55
    for lab in labels:
        data = M[lab].astype(float)
        total = int(data.sum())
        print(f"{lab}: total_trips={total}")

        if log_scale:
            data = np.log1p(data)

        plt.figure(figsize=(7, 6))
        plt.imshow(
            data,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="log(1+count)" if log_scale else "count")
        plt.xlabel("dropoff_region_id (0~54)")
        plt.ylabel("pickup_region_id (0~54)")
        plt.title(f"Pickup→Dropoff Region Heatmap (55×55) | trip_duration={lab}")
        plt.tight_layout()
        plt.show()

    return M, labels, (n_lon, n_lat)




import os
import numpy as np
import pandas as pd

def split_train_by_trip_duration(
    train_path="train.csv",
    out_dir="train_by_duration",
    chunksize=100_000,
    duration_bins=(0, 300, 600, 900, 1200, 1500, np.inf),  # 6 段
    usecols=None,
):
    """
    将 train.csv 按 trip_duration 分箱后拆分成 6 个子 CSV 文件：
    out_dir/
      train_dur_0_300.csv
      train_dur_300_600.csv
      ...
      train_dur_1500_inf.csv

    分箱规则：left-closed, right-open（[a,b)）
    """
    os.makedirs(out_dir, exist_ok=True)

    if usecols is None:
        # 你也可以把其它字段加进来，比如 id, pickup_datetime 等
        usecols = [
            "trip_duration",
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
        ]

    duration_bins = np.array(duration_bins, dtype=float)

    # 生成 6 个标签
    labels = []
    for i in range(len(duration_bins) - 1):
        a = int(duration_bins[i])
        b = duration_bins[i + 1]
        if np.isfinite(b):
            labels.append(f"{a}_{int(b)}")      # e.g. 0_300
        else:
            labels.append(f"{a}_inf")           # e.g. 1500_inf

    # 每个 label 对应一个输出路径
    out_paths = {lab: os.path.join(out_dir, f"train_dur_{lab}.csv") for lab in labels}

    # 为了追加写 CSV：记录每个文件是否已写过表头
    wrote_header = {lab: os.path.exists(out_paths[lab]) and (os.path.getsize(out_paths[lab]) > 0)
                    for lab in labels}

    total_written = {lab: 0 for lab in labels}

    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunksize):
        dur = pd.to_numeric(chunk["trip_duration"], errors="coerce")

        # 基础清洗：dur 合法
        m = dur.notna() & (dur >= 0)
        if not m.any():
            continue

        chunk = chunk.loc[m].copy()
        dur = dur.loc[m]

        # 分箱
        cat = pd.cut(
            dur.to_numpy(),
            bins=duration_bins,
            right=False,
            labels=labels,
            include_lowest=True,
        )

        # 丢弃落不到任何箱子的（一般不会）
        valid = pd.notna(cat)
        if not np.any(valid):
            continue

        chunk = chunk.iloc[np.where(valid)[0]].copy()
        cat = cat[valid]

        # 按 label 写入对应子文件（追加）
        for lab in pd.unique(cat):
            lab = str(lab)
            sub = chunk.loc[np.array(cat == lab)]
            if sub.empty:
                continue

            sub.to_csv(
                out_paths[lab],
                index=False,
                mode="a",
                header=not wrote_header[lab],
            )
            wrote_header[lab] = True
            total_written[lab] += len(sub)

    print("Done. Written rows per file:")
    for lab in labels:
        print(f"  {lab}: {total_written[lab]} rows -> {out_paths[lab]}")

    return out_paths, labels
if __name__ == "__main__":# 运行
    # 运行
    # M, labels, grid_shape = pickup_dropoff_heatmap_55x55_by_trip_duration(
    #     train_path="train.csv",
    #     chunksize=100_000,
    #     lon_range=(-74.3, -73.4),
    #     lat_range=(40.5, 41.0),
    #     n_lon=11,
    #     n_lat=5,
    #     duration_bins=(0, 300, 600, 900, 1200, 1500, np.inf),
    #     log_scale=True,
    #     vmin=0,
    #     vmax=10
    # ) 可视化
    out_paths, labels = split_train_by_trip_duration(
        train_path="train.csv",
        out_dir="train_by_duration",
        chunksize=100_000,
        duration_bins=(0, 300, 600, 900, 1200, 1500, np.inf),
    )