import random

from collections import defaultdict
import json
from tqdm import tqdm
import os
import re
TOTAL_KEYS = 40


def getFrequent_ID():
    frequent_list = []
    try:
        with open("movie_top_f.txt",'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=40, desc="正在获取key的真实频率数据")
            for line in pbar:
                index = line.strip().split(' ')[0]
                frequent_list.append(int(index))
        f.close()
    except FileNotFoundError:
        return None
    return frequent_list
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
    ID_list = getFrequent_ID()
    try:
        pattern = re.compile(r'\((\d+),([\d.]+)\)')
        with open("movie_ratings_formal.txt", 'r') as f:
            pbar = tqdm(f, total=8000, desc="正在处理用户数据")
            for line in pbar:
                matches = pattern.findall(line)
                user_keys = []
                user_values = []
                for match in matches:
                    first_num = int(match[0])  # 第一个数字转换为整数
                    second_num = float(match[1])  # 第二个数字转换为浮点数

                    user_keys.append(first_num)
                    user_values.append(second_num)
                sample_index = random.randint(0, 39)
                if ID_list[sample_index] not in user_keys:
                    users_data.append((ID_list[sample_index], -10))
                else:
                    v = user_keys.index(ID_list[sample_index])
                    users_data.append((ID_list[sample_index], user_values[v]))
    except FileNotFoundError:
        print(f"错误: 文件 movie_ratings_formal.txt 未找到。请先运行生成数据的部分。")
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


# --- 主程序入口 ---
if __name__ == "__main__":
    uniformsampling("uniform_random_sampling.txt")
