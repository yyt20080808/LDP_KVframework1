import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import json


def read_data(file_path):
    user_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(line) > 5:
                values = line.strip().split('\t')
            user_data.append(values)
        return user_data


def count_lines_in_file(filepath):
    """高效地计算文件行数，用于tqdm进度条。"""
    print(f"正在准备... 首先计算文件 '{filepath}' 的总行数。")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f)
        print(f"文件总行数: {line_count}")
        return line_count
    except FileNotFoundError:
        return 0


def calculate_key_statistics(FILE_PATH):
    """
    读取用户数据文件，计算每个键的出现频率，并计算频率的均值。
    """
    # 1. 初始化一个字典来存储每个键的频率
    # 我们预先将0-99的键都创建好，值为0
    key_frequencies = {str(i): 0 for i in range(100)}
    key_means = {str(i): 0 for i in range(100)}
    # 2. 获取文件总行数以显示进度
    num_lines = count_lines_in_file(FILE_PATH)
    if num_lines == 0:
        print(f"错误: 文件 '{FILE_PATH}' 未找到或为空。")
        return

    # 3. 逐行读取文件并更新频率
    print("正在读取所有用户数据并计算密钥频率...")
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            # 使用tqdm包装文件迭代器以显示进度条
            for line in tqdm(f, total=num_lines, desc="处理用户进度"):
                user_data = json.loads(line)

                # 遍历当前用户拥有的所有键
                for key in user_data.keys():
                    # 确保这个键在我们追踪的频率字典中
                    if key in key_frequencies:
                        key_frequencies[key] += 1
                        key_means[key] += user_data.get(key)
    except (IOError, json.JSONDecodeError) as e:
        print(f"处理文件时发生错误: {e}")
        return

    # 4. 计算频率的均值
    # 从频率字典中获取所有的计数值
    all_frequency_counts = list(key_frequencies.values())
    all_means = list(key_means.values())
    if not all_frequency_counts:
        print("没有计算出任何频率。")
        return

    all_means = [all_means[i] / all_frequency_counts[i] for i in range(100)]
    all_frequency_counts = np.array(all_frequency_counts) / 1000000
    # 5. 打印结果
    print("\n--- 计算结果 ---")
    print("已成功处理所有用户数据。")

    print("\n部分键的实际频率（示例前10个）:")
    for i in range(100):
        print(f"  - 键 '{i}': {all_frequency_counts[i]} ,mean:  {all_means[i]}")

    return all_frequency_counts, all_means

    # 理论分析与对比


def read_stastistics(FILE_PATH1, FILE_PATH2,numofusers=200000):
    real_fres = []
    real_means = []
    try:
        with open(FILE_PATH1, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=40, desc="正在获取key的真实频率数据")
            for line in pbar:
                index = line.strip().split(' ')[-1]
                real_fres.append(float(index)/numofusers)
        f.close()
    except FileNotFoundError:
        return None, None
    try:
        with open(FILE_PATH2, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=40, desc="正在获取value的真实mean数据")
            for line in pbar:
                value = line.strip().split(' ')[-1]
                real_means.append((float(value)-2.5)/5)
        f.close()
    except FileNotFoundError:
        return None, None
    return real_fres, real_means

def VPP(a, epsilon):
    p1 = (1 + a) / 2
    if random.random() < p1:
        a_star = 1
    else:
        a_star = -1
    p2 = np.exp(epsilon) / (1 + np.exp(epsilon))
    if random.random() > p2:
        a_star = -1 * a_star
    return a_star


def privkvm(epsilon, users_data, realfres, realmeans,num_index=100):
    p1 = np.exp(epsilon / 2) / (1 + np.exp(epsilon / 2))
    fre_mses, mean_mses = 0, 0
    for index in range(num_index):
        values = users_data[index]
        newValues = []
        if num_index != 100:
            for v in values:
                v = float(v)
                if v > -9:
                    newValues.append((float(v)-2.5)/5)
                else:
                    newValues.append(-10)
        count_number = 0
        count_1 = 0
        count_0 = 0
        noise_v_1 = 0
        noise_v_0 = 0
        for v in newValues:
            if random.random() < 0.3:
                continue
            count_number += 1
            v = float(v)
            if v < -9:
                if random.random() < p1:
                    count_0 += 1
                else:
                    count_1 += 1
                    if VPP(0, epsilon / 2) > 0:
                        noise_v_1 += 1
                    else:
                        noise_v_0 += 1
            else:
                if random.random() < p1:
                    count_1 += 1
                    if VPP(v, epsilon / 2) > 0:
                        noise_v_1 += 1
                    else:
                        noise_v_0 += 1
                else:
                    count_0 += 1
        fre = (count_1 / count_number + p1 - 1) / (2 * p1 - 1)
        cal_n_1 = (p1-1) / (2*p1-1)*count_1 + noise_v_1/ (2*p1-1)
        cal_n_0 = (p1-1) / (2*p1-1)*count_1 + noise_v_0/ (2*p1-1)
        if cal_n_1 > count_1:
            cal_n_1 = count_1
        elif cal_n_1 <0:
            cal_n_1 = 0
        if cal_n_0 > count_1:
            cal_n_0 = count_1
        elif cal_n_0 < 0:
            cal_n_0 = 0
        mean = (cal_n_1 - cal_n_0) / (noise_v_1+noise_v_0)
        print(f"key:{index}\tfre:{fre}\tmean:{mean}")
        fre_mses += (fre - realfres[index]) ** 2
        mean_mses += (mean - realmeans[index]) ** 2
    print(f"fre_mses:{fre_mses / num_index}")
    print(f"mean_mses:{mean_mses / num_index}")
    return fre_mses / num_index, mean_mses / num_index

if __name__ == "__main__":
    # real_fres, real_means = calculate_key_statistics("../datasets/gaussian_dataset/user_raw_data.jsonl")
    # user_data = read_data("../datasets/gaussian_dataset/uniform_random_sampling.txt")
    # privkvm(4, user_data, real_fres, real_means)

    # real_fres, real_means = calculate_key_statistics("../datasets/powerlaw_dataset/user_raw_data.jsonl")
    # user_data = read_data("../datasets/powerlaw_dataset/uniform_random_sampling.txt")
    real_fres, real_means = read_stastistics("../datasets/movie/movie_top_f.txt","../datasets/movie/movie_top_mean.txt")
    user_data = read_data("../datasets/movie/uniform_random_sampling.txt")
    res_fre, res_mean = [],[]
    for ep in [0.5, 1, 1.5, 2, 2.5, 3, 4]:
        a,b = privkvm(ep, user_data, real_fres, real_means,num_index=40) #
        res_fre.append(a)
        res_mean.append(b)
    print(res_fre)
    print(res_mean)

