import numpy as np
import xxhash
from collections import defaultdict
import json
from tqdm import tqdm
import os
import re
import random


def getdatafromfile():
    first_column_set = {}
    with open('movie_paddingSimple_3.txt', 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            b = int(col1)
            if b in first_column_set:
                first_column_set[b] += 1
            else:
                first_column_set[b] = 1
        file.close()
    print("read file ok!")
    return first_column_set


def lh_perturb(real_dist, g, p):
    n = sum(real_dist)
    noisy_samples = np.zeros(n, dtype=object)
    samples_one = np.random.random_sample(n)
    seeds = np.random.randint(0, n, n)

    counter = 0
    for k, v in enumerate(real_dist):
        k_str = str(int(k))
        for _ in range(v):
            x = xxhash.xxh32(k_str, seed=seeds[counter]).intdigest() % g
            if samples_one[counter] <= p:
                y = x
            else:
                y = np.random.randint(0, g - 1)
                if y >= x:
                    y += 1
            noisy_samples[counter] = (y, seeds[counter])
            counter += 1
    print("perturb ok!")
    return noisy_samples


def lh_aggregate(noisy_samples, domain, eps):
    p = 0.5
    g = max(3, int(np.exp(eps)) + 1)
    q = 1 / g
    n = len(noisy_samples)
    est = np.zeros(domain, dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                est[v] += 1
    # est = np.zeros(domain, dtype=np.int32)
    # seeds = np.array([sample[1] for sample in noisy_samples])
    # for v in range(domain):
    #     v_str = str(v)
    #     hashes = np.array([xxhash.xxh32(v_str, seed=seed).intdigest() % g for seed in seeds])
    #     est[v] = np.sum(hashes == np.array([sample[0] for sample in noisy_samples]))

    a = 1.0 / (p - q)
    b = n * q / (p - q)
    est = a * est - b
    print("aggregate ok!")
    return est / n


def obstain_pop_set():
    userset = getdatafromfile()
    b = userset.keys()
    real_dist = []
    index_dist = []
    epsilon = 2
    for i in range(280000):
        if i in b and userset[i] > 50:
            real_dist.append(userset[i])
            index_dist.append(i)
    print("real distribution ok!", len(real_dist))
    g = max(3, int(np.exp(epsilon)) + 1)
    noisy_samples = lh_perturb(real_dist, g, 1 / 2)
    est = lh_aggregate(noisy_samples, len(real_dist), epsilon)
    k_pop_IDs = []
    for k, v in enumerate(est):
        if v > 0.005:
            k_pop_IDs.append(index_dist[k])
    print(k_pop_IDs)
    return k_pop_IDs


def read_l(pop_ids):  # 获取 用户的 l 的分布
    id_set = set(pop_ids)
    l_numbers = []
    try:
        pattern = re.compile(r'\((\d+),([\d.]+)\)')
        with open("movie_ratings_formal.txt", 'r') as f:
            pbar = tqdm(f, total=200000, desc="正在处理用户数据")
            for line in pbar:
                matches = pattern.findall(line)
                l = 0
                for match in matches:
                    first_num = int(match[0])  # 第一个数字转换为整数
                    # second_num = float(match[1])  # 第二个数字转换为浮点数
                    if first_num in id_set:
                        l += 1
                l_numbers.append(l)
            f.close()
        from collections import Counter
        # 统计频率
        counter = Counter(l_numbers)
        print(counter)
    except FileNotFoundError:
        print(f"错误: 文件 movie_ratings_formal.txt 未找到。请先运行生成数据的部分。")
        return None
    l_numbers.sort()
    n = len(l_numbers)
    index = int(0.75 * n) - 1  # 下标（从 0 开始）
    percentile_90 = l_numbers[index]
    return percentile_90

def random_sample(a, k):
    if k > len(a):
        raise ValueError("k cannot be greater than the length of the list a")
    return random.sample(a, k)
def padding_sampling_from_popularset(l=20, pop_ids=[1]):
    id_set = set(pop_ids)
    dummy_list = [(v, 0) for v in range(14000000, 14000000 + l)]
    sampled_values = []
    try:
        pattern = re.compile(r'\((\d+),([\d.]+)\)')
        with (open("movie_ratings_formal.txt", 'r') as f):
            pbar = tqdm(f, total=200000, desc="正在处理用户数据")
            for line in pbar:
                matches = pattern.findall(line)
                user_keys = []
                user_values = []
                for match in matches:
                    first_num = int(match[0])  # 第一个数字转换为整数
                    second_num = float(match[1])  # 第二个数字转换为浮点数
                    if first_num in id_set:
                        user_keys.append(first_num)
                        user_values.append(second_num)
                new_list = [(user_keys[i], user_values[i]) for i in range(len(user_keys))]
                if len(new_list) > l:
                    random.shuffle(new_list)
                    new_list = new_list[0:l]
                elif len(new_list) < l:
                    new_list.extend(random_sample(dummy_list, l - len(new_list)))
                sampled_value = np.random.randint(0, l, 1)[0]
                sampled_values.append(new_list[sampled_value])
            f.close()
    except FileNotFoundError:
        print(f"错误: 文件 movie_ratings_formal.txt 未找到。请先运行生成数据的部分。")
        return None

    with open('movie_paddingSimple_Ours.txt', 'w') as file:
        for sampled_value in sampled_values:
            file.write(f'{sampled_value[0]},{sampled_value[1]}\n')
        file.close()


if __name__ == "__main__":
    # k_pop_IDs = obstain_pop_set()

    k_pop_IDs = [1, 47, 50, 150, 216, 231, 256, 260, 296, 318, 349, 356, 480, 527, 593, 628, 661, 788, 858, 899, 919,
                 953, 1036, 1073, 1210, 1235, 1617, 1721, 1884, 1917, 1923, 1968, 2020, 2571, 2694, 2871, 2959, 2997,
                 3555, 4226, 4239, 4306, 4993, 5464, 5952, 7153, 8368, 8464, 30749, 48385, 53125, 56367, 57669, 79132,
                 79702, 106487]
    # l_star =  read_l(k_pop_IDs)
    # print(l_star)
    l_star = 10
    padding_sampling_from_popularset(l=l_star, pop_ids=k_pop_IDs)
