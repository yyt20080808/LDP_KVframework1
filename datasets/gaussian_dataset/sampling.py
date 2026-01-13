import random

from collections import defaultdict
import json
from tqdm import tqdm
import os

FILE_PATH = "user_raw_data.jsonl"
TOTAL_KEYS = 100


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




# --- 主程序入口 ---


def uniformsampling(output_txt_path):
    """
    从 JSON Lines 文件中读取数据。
    """
    # print(f"\n--- 从 {FILE_PATH} 中读取数据 (前 {num_to_read} 个用户示例) ---")
    users_data = []
    try:
        with open(FILE_PATH, 'r') as f:
            pbar = tqdm(f, total=1000000, desc="正在处理用户数据")
            for line in pbar:
                # if i >= num_to_read:
                #     break
                user_data = json.loads(line.strip())
                sample_index = random.randint(0, 49)
                key_to_index = str(sample_index)
                if key_to_index not in user_data:
                    users_data.append((sample_index, -10))
                else:
                    users_data.append((sample_index, user_data.get(key_to_index)))
    except FileNotFoundError:
        print(f"错误: 文件 {FILE_PATH} 未找到。请先运行生成数据的部分。")
        return None
    grouped_data = defaultdict(list)
    # 2. 遍历原始数据
    for item_tuple in users_data:
        # item_tuple[0] 是键 (例如 10)
        # 直接将整个元组追加到对应键的列表中
        # 如果键是第一次遇到，defaultdict 会自动创建 grouped_data[10] = []
        grouped_data[item_tuple[0]].append(item_tuple[1])

    # grouped_data 现在是一个字典，其值是聚合后的列表

    print(f"正在将数据以人类可读格式保存到: {output_txt_path}")
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # 为了输出好看，可以按键排序
            sorted_keys = sorted(grouped_data.keys())

            for key in sorted_keys:
                value_list = grouped_data[key]
                # 写入键
                # f.write(f"\nKey: {key}\n")

                # 逐行写入该键对应的元组列表
                for item_tuple in value_list:
                    f.write(f"\t{item_tuple}")
                f.write("\n")
                # 添加分隔符，让文件更清晰
                # f.write("-" * 20 + "\n")

        print("文件保存成功！")

    except IOError as e:
        print(f"文件写入时发生错误: {e}")

def padding_and_sampling(output_txt_path, fill_len=7, pad_start=50, missing_value=-10):
    """
    每行：从存在的 key 中取候选；不足 fill_len 用 50,51,... 补齐；再随机抽一个。
    存储与原代码一致：grouped_data[chosen_index].append(value)
    """
    users_data = []

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=1000000, desc="正在处理用户数据(填充+采样)")
            for line in pbar:
                user_data = json.loads(line.strip())

                # 1) 收集该行实际存在的 key（只统计 0~49）
                existing_indices = []
                for i in range(TOTAL_KEYS):
                    if str(i) in user_data:
                        existing_indices.append(i)

                # 2) 填充到 fill_len（用 50,51,52...）
                candidates = existing_indices[:]
                next_pad = pad_start
                while len(candidates) < fill_len:
                    candidates.append(next_pad)
                    next_pad += 1

                chosen_index = random.choice(candidates)

                if chosen_index >= pad_start:
                    chosen_value = missing_value
                else:
                    chosen_value = user_data.get(str(chosen_index), missing_value)

                users_data.append((chosen_index, chosen_value))

    except FileNotFoundError:
        print(f"错误: 文件 {FILE_PATH} 未找到。")
        return None

    grouped_data = defaultdict(list)
    for idx, val in users_data:
        grouped_data[idx].append(val)

    print(f"loading and : {output_txt_path}")
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for key in sorted(grouped_data.keys()):
                for v in grouped_data[key]:
                    f.write(f"\t{v}")
                f.write("\n")
        print("文件保存成功！")
    except IOError as e:
        print(f"文件写入时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # uniformsampling("uniform_random_sampling.txt")
    padding_and_sampling("padding_and_sampling.txt",fill_len=11, pad_start=100, missing_value=10)

