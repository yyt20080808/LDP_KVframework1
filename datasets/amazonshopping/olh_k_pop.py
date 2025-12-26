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
    with open('amazon_paddingSimple_2.txt', 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            b = str(col1)
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
    epsilon = 10
    for i in b:
        if userset[i] > 200:
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
        pattern = re.compile(r'\(([\da-zA-Z]+),([\d.]+)\)')
        with open("amazon_ratings_formal.txt", 'r') as f:
            pbar = tqdm(f, total=200000, desc="正在处理用户数据")
            for line in pbar:
                matches = pattern.findall(line)
                l = 0
                for match in matches:
                    first_num = match[0]  # 第一个数字转换为整数
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
    print(sum(l_numbers))
    # print(l_numbers)
    p90 = np.percentile(l_numbers, 90)
    percentile_90 = p90
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
        pattern = re.compile(r'\(([A-Za-z0-9]+),([\d.]+)\)')
        with (open("amazon_ratings_formal.txt", 'r') as f):
            pbar = tqdm(f, total=200000, desc="正在处理用户数据")
            for line in pbar:
                matches = pattern.findall(line)
                user_keys = []
                user_values = []
                for match in matches:
                    first_num = match[0]  # 第一个数字转换为整数
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

    with open('amazon_paddingSimple_Ours.txt', 'w') as file:
        for sampled_value in sampled_values:
            file.write(f'{sampled_value[0]},{sampled_value[1]}\n')
        file.close()


if __name__ == "__main__":
    k_pop_IDs = obstain_pop_set()

    k_pop_IDs = ['B000C1UFP2', 'B000ASDGK8', 'B005UBN5AQ', 'B000UVZU1S', 'B004Z40048', 'B001AO0WCG', 'B002X9XV48',
                 'B0046VGPHQ', 'B004DK0UDA', 'B000C1ZDTU', 'B0007IQMVG', 'B008B4FKLG', 'B003H146PY', 'B00005O0MZ',
                 'B008CEDY5O', 'B001MA0QY2', 'B002WTC38O', 'B00147FGJ8', 'B000NSQI4Q', 'B0056GDG90', 'B0068Y6CA4',
                 'B000F63TW0', 'B004D24818', 'B00004TUBL', 'B004OHQR1Q', 'B007BLN17K', 'B009GIOVKC', 'B001RMP7M6',
                 'B000ELP5KA', 'B003BQ6QXK', 'B000X2STQ2', 'B002DOO1VA', 'B003MJ7Z0O', 'B0047ETXD0', 'B00121UVU0',
                 'B000F35R00', 'B0069FDR96', 'B000PY17PI', 'B001ET77NY', 'B000142FVW', 'B000U0F9GA', 'B0000CC64W',
                 'B0009PVV40', 'B000FS05VG', 'B003V265QW', 'B005XIDZHO', 'B003S516XO', 'B00A51LI1O', 'B000EPJNMW',
                 'B003FBI9LS', 'B002WTC37A', 'B006SVCY6I', 'B003QKL5YQ', 'B000TKH6G2', 'B0002JKPA4', 'B005X2F7KI',
                 'B000GHWSG6', 'B00A76CPUA', 'B0043OYFKU', 'B00150LT40', 'B006U98T08', 'B004YBW5T0', 'B002UU9Q6W',
                 'B002JSL6QI', 'B0002Z8HAI', 'B0012J30LY', 'B008O4YM4Y', 'B0030O3VRW', 'B000L596FE', 'B005GHP5UC',
                 'B000CC08VC', 'B003COAFPQ', 'B002TPQPEE', 'B008LDFU26', 'B000ZMBSPE', 'B002RXW5BA', 'B000B8FW0Y',
                 'B0009FHJRS', 'B0043494XS', 'B000VPPUEA', 'B0007W1R58', 'B00006IV2F', 'B00538TSMU', 'B00021DVCQ',
                 'B0067JF88M', 'B004BHAJAE', 'B0000632EN', 'B002LB75AO', 'B0000UTUW4', 'B002MSN3QQ', 'B006Z96OI2',
                 'B009HULKLW', 'B0000YUXI0', 'B0002Z8QG8', 'B00011QUDE', 'B003F2T0M4', 'B00021C1LI', 'B006ZBP8NM',
                 'B000PLUZL8', 'B003V21WO2', 'B000WNLFBI', 'B0048O2R1E', 'B005R3IPJI', 'B001WAKUTS', 'B004Z209HS',
                 'B0009V1YR8', 'B0009V8N5E', 'B00GP184WO', 'B008U12YV4', 'B00016XJ4M', 'B000NWGCZ2', 'B001A3ML3K',
                 'B000OYJ9AO', 'B005OSQGN8', 'B0090UJFYI', 'B004VFXVJW', 'B003UNP20W', 'B000R80ZTQ', 'B005UBN2AO',
                 'B00639DLV2', 'B00132ZG3U', 'B0001ZYLAO', 'B000ICRDSW', 'B00021AK4I', 'B000IOFQWK', 'B008MP481M',
                 'B000ODNSR0', 'B001HTYJLO', 'B003ILUQPM', 'B009T47YZ2', 'B000P22TIY', 'B001CT0AGC', 'B0009OAGXI',
                 'B0000530ED', 'B000OZDOCW', 'B000ZLVUYO', 'B00IALDHDI', 'B005BF1M10', 'B001JKTTVQ', 'B001U9M2EW',
                 'B0008IV7BU', 'B003UH0528', 'B000UM2KCY', 'B002MZ8BK2', 'B002QANC2A', 'B002GCKVJA', 'B004XA81ZE',
                 'B008RVYJS8', 'B000S0CKRS', 'B006K9OQSC', 'B0047PPCAW', 'B002CSPKSU', 'B004TSFBNK', 'B000VDUOFM',
                 'B000EVGQ0S', 'B006L1DNWY', 'B000HRVC5I', 'B004GQZX4M', 'B00325D0WK', 'B0000ZLEFU', 'B00C7DYBX0',
                 'B003V264WW', 'B00GS83884', 'B001330XFA', 'B003M6AS4C', 'B000UPRSKA', 'B004INUWX0', 'B0009XH6TG',
                 'B0000Y3LAC', 'B00188IFHS', 'B00I073WLE', 'B00067YSLO', 'B0085WHBHU', 'B00FAEOCP0', 'B0058E3XJI',
                 'B0077CNO1G', 'B000RRKCVS', 'B0012BNVE8', 'B000F5AG5E', 'B006T6Y56E', 'B0002VQ0WO', 'B00CFRBIC0',
                 'B009GYVMAS', 'B004WSXD4G', 'B001MWV40U', 'B0000AFUTL', 'B006IBM21K', 'B002JPJ2ZS', 'B000142C1A',
                 'B009RNUH4A', 'B002VLZHLI', 'B007Q0WW0S', 'B005Y6F4WO', 'B0009OAHC8', 'B001UHN0I6', 'B004FEKA3E',
                 'B007TNHQOY', 'B00912CL5K', 'B000Q8LFX2', 'B004LJ0ZK6', 'B00E68O4JU', 'B008U1Q4DI', 'B001BALMCS',
                 'B0073TX6IO', 'B008TBTA6C', 'B00176B9JC', 'B0006Q3NTS', 'B000NNDNYY', 'B000OQ2DL4', 'B00D6EDGYE',
                 'B0079R6BD2', 'B00CNOUZE2', 'B006QO4BRM', 'B009GEUPDS', 'B0062MAAAK', 'B00D9NV20C', 'B00AM8923G',
                 'B009SCRSHE', 'B008HODSNW', 'B0067F8BBM', 'B008GOR6O0', 'B0070WVEWE', 'B00L5JHZJO']

    # l_star =  read_l(k_pop_IDs)
    # print(l_star)
    # l_star = 1
    # padding_sampling_from_popularset(l=l_star, pop_ids=k_pop_IDs)
