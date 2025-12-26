import numpy as np
import json
from tqdm import tqdm
import os

# --- 1. 参数定义 ---
NUM_USERS = 1_000_000
TOTAL_KEYS = 100

# 键的选择倾斜程度（越大越集中在少数 key 上）
KEY_SELECTION_EXPONENT = 1.1

# 值的分布（越大越靠近 1）
VALUE_POWER_LAW_EXPONENT = 4.0

# 你要的 keys 数量分布目标
MIN_KEYS_PER_USER = 1
MAX_KEYS_PER_USER = 6
TARGET_MEAN_KEYS = 3.0

FILE_PATH = "user_raw_data.jsonl"


def calibrate_num_keys_distribution(
    target_mean=3.0,
    max_k=6,
    n_probe=300_000,
    a_grid=None,
    p1_grid=None,
    seed=42
):
    """
    自动校准:
    num_keys = 1 (with prob p1)
    else num_keys = min(max_k, zipf(a) + 1)  # 2..max_k, 且 mode 往往是 2

    目标：
    - mean 接近 target_mean
    - 同时尽量让 P(num_keys=2) 更大（符合“大部分用户只有2个数据”）
    """
    rng = np.random.default_rng(seed)

    if a_grid is None:
        a_grid = np.linspace(1.2, 3.5, 60)   # Zipf 指数搜索范围
    if p1_grid is None:
        p1_grid = np.linspace(0.00, 0.25, 51) # 取 1 的概率搜索范围

    best = None  # (score, a, p1, mean, p2, probs)

    for a in a_grid:
        # 先一次性生成 zipf 样本，避免每次内层循环重复采样（更快）
        z = rng.zipf(a, size=n_probe)
        k_base = np.minimum(max_k, z + 1)  # 2..max_k

        for p1 in p1_grid:
            u = rng.random(n_probe)
            k = np.where(u < p1, 1, k_base)

            mean_k = k.mean()
            p2 = (k == 2).mean()

            # 评分：优先满足均值，然后尽量提升 p2
            mean_error = abs(mean_k - target_mean)
            score = mean_error * 10.0 - p2  # 均值误差权重大一些

            if best is None or score < best[0]:
                # 统计 1..max_k 的概率（用于你检查形状）
                probs = np.array([(k == i).mean() for i in range(1, max_k + 1)])
                best = (score, a, p1, mean_k, p2, probs)

    _, best_a, best_p1, best_mean, best_p2, best_probs = best
    return best_a, best_p1, best_mean, best_p2, best_probs


def sample_num_keys(rng, a, p1, max_k=6):
    """按上面的校准分布采样单个用户的 key 数量，范围 1..max_k"""
    if rng.random() < p1:
        return 1
    return int(min(max_k, rng.zipf(a) + 1))


def generate_and_save_data():
    print(f"正在校准 num_keys 分布，使均值≈{TARGET_MEAN_KEYS}，且最多不超过 {MAX_KEYS_PER_USER} ...")
    best_a, best_p1, best_mean, best_p2, best_probs = calibrate_num_keys_distribution(
        target_mean=TARGET_MEAN_KEYS,
        max_k=MAX_KEYS_PER_USER
    )

    print("校准完成：")
    print(f"  Zipf exponent a = {best_a:.4f}")
    print(f"  P(num_keys=1) p1 = {best_p1:.4f}")
    print(f"  实际均值 mean = {best_mean:.4f}")
    print(f"  P(num_keys=2) = {best_p2:.4f}")
    print("  1..6 概率分布 =", {i+1: float(best_probs[i]) for i in range(MAX_KEYS_PER_USER)})

    print(f"\n正在生成 {NUM_USERS} 个用户的数据并保存至 {FILE_PATH}...")

    rng = np.random.default_rng(123)

    # --- key 的幂律选择概率（越小 rank 越容易被选中） ---
    possible_keys_array = np.arange(TOTAL_KEYS)
    key_ranks = possible_keys_array + 1
    weights = 1 / (key_ranks ** KEY_SELECTION_EXPONENT)
    key_selection_probabilities = weights / np.sum(weights)

    with open(FILE_PATH, 'w') as f:
        for _ in tqdm(range(NUM_USERS), desc="生成进度"):
            num_keys_for_user = sample_num_keys(
                rng=rng, a=best_a, p1=best_p1, max_k=MAX_KEYS_PER_USER
            )

            user_keys = rng.choice(
                possible_keys_array,
                size=num_keys_for_user,
                replace=False,
                p=key_selection_probabilities
            )

            user_values = rng.power(VALUE_POWER_LAW_EXPONENT, size=num_keys_for_user)
            user_values = np.clip(user_values, -1.0, 1.0)

            user_data = {str(int(k)): float(v) for k, v in zip(user_keys, user_values)}
            f.write(json.dumps(user_data) + '\n')

    print("数据生成和存储完成。")
    file_size = os.path.getsize(FILE_PATH) / (1024 * 1024)
    print(f"生成的文件大小: {file_size:.2f} MB")


if __name__ == "__main__":
    generate_and_save_data()
