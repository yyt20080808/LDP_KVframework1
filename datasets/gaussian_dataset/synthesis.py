import numpy as np
import json
from tqdm import tqdm
import os

# --- 1. 参数定义 ---
NUM_USERS = 1_000_000
TOTAL_KEYS = 100
KEYS_MEAN = 11.3
KEYS_STD = 0.4
VALUE_MEAN = 0.3
VALUE_STD = 0.1
FILE_PATH = "user_raw_data.jsonl"


def generate_and_save_data():
    """
    生成合成数据并将其逐行保存到 JSON Lines 文件中。
    """
    print(f"正在生成 {NUM_USERS} 个用户的数据并保存至 {FILE_PATH}...")

    # 使用 'w' 模式打开文件，如果文件已存在则会覆盖
    with open(FILE_PATH, 'w') as f:
        # 使用 tqdm 显示进度条
        for _ in tqdm(range(NUM_USERS), desc="生成进度"):
            # --- 为单个用户生成数据 ---

            # 2.1. 决定该用户拥有多少个键值对
            # 从高斯分布中抽样，并确保键的数量是整数且在合理范围内
            num_keys_for_user = int(round(np.random.normal(KEYS_MEAN, KEYS_STD)))
            num_keys_for_user = max(0, min(TOTAL_KEYS, num_keys_for_user))  # 保证数量在 [0, 100] 之间

            # 2.2. 确定是哪些键 (从 0-99 中无重复地选择)
            possible_keys = range(TOTAL_KEYS)
            user_keys = np.random.choice(possible_keys, size=num_keys_for_user, replace=False)

            # 2.3. 为每个键生成对应的值
            user_values = np.random.normal(VALUE_MEAN, VALUE_STD, size=num_keys_for_user)

            # 2.4. 将值裁剪到 [-1, 1] 区间
            user_values = np.clip(user_values, -1.0, 1.0)

            # 2.5. 创建用户数据的字典
            user_data = {str(k): v for k, v in zip(user_keys, user_values)}

            # 2.6. 将用户数据写入文件，每个用户占一行
            f.write(json.dumps(user_data) + '\n')

    print("数据生成和存储完成。")
    file_size = os.path.getsize(FILE_PATH) / (1024 * 1024)
    print(f"生成的文件大小: {file_size:.2f} MB")


def read_data(num_to_read=5):
    """
    从 JSON Lines 文件中读取数据。

    Args:
        num_to_read (int): 要读取并打印的用户数量。
    """
    print(f"\n--- 从 {FILE_PATH} 中读取数据 (前 {num_to_read} 个用户示例) ---")
    users_data = []
    try:
        with open(FILE_PATH, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_to_read:
                    break
                user_data = json.loads(line.strip())
                users_data.append(user_data)
                print(f"用户 {i + 1}: {user_data}")
        return users_data
    except FileNotFoundError:
        print(f"错误: 文件 {FILE_PATH} 未找到。请先运行生成数据的部分。")
        return None


# --- 主程序入口 ---
if __name__ == "__main__":
    # 步骤1: 生成并存储数据
    generate_and_save_data()

    # 步骤2: 读取数据进行验证
    read_data()