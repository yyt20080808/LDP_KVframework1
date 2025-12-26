import csv

# 读取CSV文件并组织数据结构
user_ratings = {}
with open('ratings.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行

    for row in reader:
        user_id = int(row[0])
        movie_id = int(row[1])

        # 处理评分格式：整数评分不带小数点
        rating = float(row[2])
        if rating.is_integer():
            rating = int(rating)

        # 将数据存入字典
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        user_ratings[user_id].append((movie_id, rating))

# 写入新TXT文件
with open('movie_ratings_formal.txt', 'w', encoding='utf-8') as txtfile:
    for user_id in sorted(user_ratings.keys()):
        # 生成元组字符串列表
        tuples = [f"({mid},{rate})" for mid, rate in user_ratings[user_id]]
        # 组合成完整行
        line = f"{user_id} " + " ".join(tuples)
        txtfile.write(line + "\n")