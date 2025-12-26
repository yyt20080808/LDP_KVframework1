import pandas as pd
import numpy as np
import random
from collections import defaultdict,Counter
import re
# Step 1: Define the path to the CSV file
file_path1 = 'test.csv'
file_path2 = 'train.csv'

# 2083778 records in totoal
# passenger_count = 1: 1476987
# passenger_count = 2: 300345
# passenger_count = 3: 85582
# passenger_count = 4: 40421
# passenger_count = 5: 111499
# passenger_count = 6: 68854
import pandas as pd

def taxi_infor():
    # needed
    usecols = [
        "pickup_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    # 1) passenger_count 分布：Counter 更适合做“全量累计”
    passenger_counter = Counter()

    # 2) 总记录数（train + test）
    total_rows = 0

    # 3) (pickup_longitude, pickup_latitude) 保留 1 位小数后的组合频次
    pickup_loc_counter = Counter()

    def process_one_file(path):
        nonlocal total_rows, passenger_counter, pickup_loc_counter

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=10_0000):
            total_rows += len(chunk)

            # --- (1) passenger_count 统计 ---
            # 注意：有些数据 passenger_count 可能是空/浮点/字符串，统一转数字并丢掉 NaN
            pc = pd.to_numeric(chunk["passenger_count"], errors="coerce").dropna().astype(int)
            passenger_counter.update(pc.value_counts().to_dict())

            # --- (3) pickup 坐标保留 1 位小数后组合频次 ---
            plon = pd.to_numeric(chunk["pickup_longitude"], errors="coerce")
            plat = pd.to_numeric(chunk["pickup_latitude"], errors="coerce")
            tmp = pd.DataFrame({"plon": plon, "plat": plat}).dropna()

            # 保留 1 位小数（四舍五入）；如果你想“截断”而不是四舍五入，我也可以给你另一版
            tmp["plon1"] = tmp["plon"].round(1)
            tmp["plat1"] = tmp["plat"].round(1)

            # 组合成 tuple 统计频次
            pickup_loc_counter.update(zip(tmp["plon1"], tmp["plat1"]))

    # 分别处理 train / test
    process_one_file(file_path1)
    process_one_file(file_path2)

    # ========== 输出结果 ==========
    print("==== (2) 总记录数 (train + test) ====")
    print(f"Total rows: {total_rows}")

    print("\n==== (1) passenger_count 频次分布 ====")
    # 按 passenger_count 从小到大打印
    for k in sorted(passenger_counter.keys()):
        print(f"passenger_count = {k}: {passenger_counter[k]}")

    print(f"\n不同 passenger_count 取值数量: {len(passenger_counter)}")

    print("\n==== (3) pickup 坐标(保留1位小数)组合频次 TopK ====")
    top_pickups = pickup_loc_counter.most_common(40)
    for (lon1, lat1), c in top_pickups:
        print(f"({lon1:.1f}, {lat1:.1f}) -> {c}")

    print(f"\npickup (lon,lat) 组合总种类数: {len(pickup_loc_counter)}")

    # 如果你后续要拿这些结果继续分析，返回它们
    # return passenger_counter, total_rows, pickup_loc_counter



PAIR_RE = re.compile(r'\(([A-Za-z0-9]+),([\d.]+)\)')

def pad_and_uniform_sample(
    input_path: str,
    output_path: str,
    l: int,
    dummy_prefix: str = "DUMMY",
    dummy_rating: float = 0.0,
    seed: int | None = None,
):
    """
    每行一个用户：
    1) 解析出 (item, rating) 列表
    2) padding / 截断到长度 l（不足补 dummy，超出随机取 l 个）
    3) 在长度 l 的列表中均匀随机采样 1 个 (item, rating)
    4) 写出：user_id,item,rating
    """
    if seed is not None:
        random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            # user_id 是行首第一个字段
            parts = line.split(maxsplit=1)
            user_id = parts[0]

            pairs = [(m.group(1), float(m.group(2))) for m in PAIR_RE.finditer(line)]

            # 如果这一行没有任何 pair，按你的业务可以跳过或全 dummy
            if not pairs:
                padded = [(f"{dummy_prefix}_{user_id}_{i}", dummy_rating) for i in range(l)]
            else:
                # 截断 / padding 到长度 l
                if len(pairs) > l:
                    padded = random.sample(pairs, l)  # 均匀随机选 l 个
                elif len(pairs) < l:
                    padded = pairs[:]
                    need = l - len(padded)
                    padded.extend([(f"{dummy_prefix}_{user_id}_{i}", dummy_rating) for i in range(need)])
                else:
                    padded = pairs

            # 均匀随机采样 1 个（dummy 也可能被采到，这是“padding 后均匀采样”的定义）
            item_id, rating = random.choice(padded)

            fout.write(f"{item_id},{rating}\n")

    print(f"Done. Wrote to: {output_path}")
def generateData():
    data = [
        'B001MA0QY2'
    ]#3.568839  2869
    first_column_set = set(data)
    data_dict = defaultdict(list)
    with open('amazon_paddingSimple100.txt', 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            if col1 in first_column_set:
                data_dict[col1].append(float(col2))
        file.close()
    print(data_dict.get('B001MA0QY2'))
    # print(data_dict.keys(), data_dict.values())


# generateData()

taxi_infor()
# pad_and_uniform_sample("amazon_ratings_formal.txt", "amazon_paddingSimple_5.txt",5)