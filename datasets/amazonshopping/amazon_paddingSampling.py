import pandas as pd
import numpy as np
import random
from collections import defaultdict
import re
# Step 1: Define the path to the CSV file
file_path = 'raw_sets/ratings_Beauty.csv'




def infor():
    # Initialize aggregators
    movie_id_counts = {}
    user_ids = set()
    ratings_sum = {}
    ratings_count = {}
    count = 0
    # Read the CSV file in chunks
    chunksize = 100000  # Adjust the chunk size based on your memory constraints
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        count += 1
        # Aggregate movie_id counts
        for movie_id in chunk['ProductId']:
            if movie_id in movie_id_counts:
                movie_id_counts[movie_id] += 1
            else:
                movie_id_counts[movie_id] = 1

        # Aggregate user_ids
        for user_id in chunk['UserId']:
            if user_id not in user_ids:
                user_ids.add(user_id)

        # Aggregate ratings sum and count for mean calculation
        for movie_id, rating in zip(chunk['ProductId'], chunk['Rating']):
            if movie_id in ratings_sum:
                ratings_sum[movie_id] += rating
                ratings_count[movie_id] += 1
            else:
                ratings_sum[movie_id] = rating
                ratings_count[movie_id] = 1

    # Step 2: Calculate the biggest number in the "movieId" column
    movie_numbers = len(movie_id_counts.keys())
    # 去除出现频率为 1 的 ProductId
    filtered_movie_id_counts = {mid: c for mid, c in movie_id_counts.items() if c > 1}

    # 取出后还剩多少种 ProductId
    num_movies_after_filter = len(filtered_movie_id_counts)
    print(f"去除频率=1 后，ProductId 还剩多少种: {num_movies_after_filter},原来有 {movie_numbers}种")

    # print(f"The biggest number in the 'ProductId' column is: {movie_numbers}")

    # Step 3: Calculate the mean of 'rating' for each 'movieId'
    mean_ratings = {movie_id: ratings_sum[movie_id] / ratings_count[movie_id] for movie_id in ratings_sum}
    mean_ratings_series = pd.Series(mean_ratings)
    # print("\nMean ratings for each ProductId:")
    # print(mean_ratings_series)

    # Step 4: Calculate the frequency of 'movieId' and get the top 40 frequent movieId
    top_40_frequent_movie_ids = pd.Series(movie_id_counts).nlargest(40)
    # print("\nTop 40 frequent ProductId and their frequency:")
    # print(top_40_frequent_movie_ids)

    # Calculate the mean ratings for the top 40 frequent movieIds
    top_40_mean_ratings = mean_ratings_series[top_40_frequent_movie_ids.index]
    # print("\nMean ratings for the top 40 frequent ProductId:")
    # print(top_40_mean_ratings)

    # Step 5: Calculate the number of unique userId
    num_unique_users = len(user_ids)
    print(f"\nNumber of unique userId: {num_unique_users}", count)

    total_freq_sum = sum(movie_id_counts.values())
    print(f"所有 ProductId 的频率之和（总 records）: {total_freq_sum}")


def random_sample(a, k):
    if k > len(a):
        raise ValueError("k cannot be greater than the length of the list a")
    return random.sample(a, k)

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

# infor()
pad_and_uniform_sample("amazon_ratings_formal.txt", "amazon_paddingSimple_5.txt",5)