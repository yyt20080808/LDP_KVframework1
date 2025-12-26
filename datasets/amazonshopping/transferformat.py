from collections import defaultdict
import csv

# 文件路径 - 替换为你的实际文件路径
input_file = 'raw_sets/ratings_Beauty.csv'
output_file = 'amazon_ratings_formal.txt'

# 初始化映射字典和计数器
user_id_map = {}  # 原始用户ID -> 新用户ID
# product_id_map = {}  # 原始产品ID -> 新产品ID
user_ratings = defaultdict(list)  # 用户ID -> [(产品ID, 评分)]

# 下一个可用的新ID
new_user_id = 1

# 读取CSV文件
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    # 处理每一行数据
    for row in reader:
        # 处理用户ID映射
        if row['UserId'] not in user_id_map:
            user_id_map[row['UserId']] = new_user_id
            new_user_id += 1
        current_user_id = user_id_map[row['UserId']]

        # 处理产品ID映射
        current_product_id = row['ProductId']

        # 添加评分记录
        user_ratings[current_user_id].append((current_product_id, row['Rating']))

# 写入到输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    # 按用户ID排序
    for user_id in sorted(user_ratings.keys()):
        # 创建该用户的行数据
        line_parts = [str(user_id)]
        # 添加所有产品ID和评分对
        for product_id, rating in user_ratings[user_id]:
            line_parts.extend(["("+str(product_id)+","+ str(rating)+")"])
        # 写入文件
        outfile.write(' '.join(line_parts) + '\n')

print(f"文件已成功生成：{output_file}")