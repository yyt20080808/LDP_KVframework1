import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Step 1: Define the path to the CSV file
file_path = './ratings.csv'


# 200948 多条记录
# 用户数目 7120
# frequency > 50% 的电影不少

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
        for movie_id in chunk['movieId']:
            if movie_id in movie_id_counts:
                movie_id_counts[movie_id] += 1
            else:
                movie_id_counts[movie_id] = 1

        # Aggregate user_ids
        for user_id in chunk['userId']:
            if user_id not in user_ids:
                user_ids.add(user_id)

        # Aggregate ratings sum and count for mean calculation
        for movie_id, rating in zip(chunk['movieId'], chunk['rating']):
            if movie_id in ratings_sum:
                ratings_sum[movie_id] += rating
                ratings_count[movie_id] += 1
            else:
                ratings_sum[movie_id] = rating
                ratings_count[movie_id] = 1

    # Step 2: Calculate the biggest number in the "movieId" column
    biggest_movie_id = max(movie_id_counts.keys())
    filtered_movie_id_counts = {mid: c for mid, c in movie_id_counts.items() if c > 1000}
    num_movies_after_filter = len(filtered_movie_id_counts)
    print(f"去除频率=1 后，ProductId 还剩多少种: {num_movies_after_filter}")
    print(f"The biggest number in the 'movieId' column is: {biggest_movie_id}")

    # Step 3: Calculate the mean of 'rating' for each 'movieId'
    mean_ratings = {movie_id: ratings_sum[movie_id] / ratings_count[movie_id] for movie_id in ratings_sum}
    mean_ratings_series = pd.Series(mean_ratings)
    print("\nMean ratings for each movieId:")
    print(mean_ratings_series)

    # Step 4: Calculate the frequency of 'movieId' and get the top 40 frequent movieId
    top_40_frequent_movie_ids = pd.Series(movie_id_counts).nlargest(60)
    print("\nTop 40 frequent movieId and their frequency:")
    print(top_40_frequent_movie_ids)

    # Calculate the mean ratings for the top 40 frequent movieIds
    top_40_mean_ratings = mean_ratings_series[top_40_frequent_movie_ids.index]
    print("\nMean ratings for the top 40 frequent movieId:")
    print(top_40_mean_ratings)

    # Step 5: Calculate the number of unique userId
    num_unique_users = len(user_ids)
    print(f"\nNumber of unique userId: {num_unique_users}", count)

    total_freq_sum = sum(movie_id_counts.values())
    print(f"所有 ProductId 的频率之和（总 records）: {total_freq_sum}")


def random_sample(a, k):
    if k > len(a):
        raise ValueError("k cannot be greater than the length of the list a")
    return random.sample(a, k)


def paddingSampling(l=100):
    # Step 1: Read the CSV file
    import pandas as pd

    # Step 1: Read the CSV file
    file_path = 'ratings.csv'
    df = pd.read_csv(file_path)

    # Initialize dummy number starting value
    dummy_list = [(v, 0) for v in range(14000000, 14000000 + l)]

    # Step 2: Create lists of length 100 for each unique user ID
    user_movie_lists = {}
    unique_users = df['userId'].unique()

    # Group by userId to reduce the number of DataFrame filter operations
    grouped = df.groupby('userId')

    for user_id in unique_users:
        # Get all movieIds for the user

        user_group = grouped.get_group(user_id)  # Retrieve the group for the user once
        user_movies = user_group['movieId'].tolist()
        user_ratings = user_group['rating'].tolist()

        new_list = [(user_movies[i], user_ratings[i]) for i in range(len(user_movies))]

        # If more than 100 movies, randomly discard to make it 100
        if len(new_list) > l:
            random.shuffle(new_list)
            new_list = new_list[0:l]
            # here is random,
        # If fewer than 100 movies, add dummy values
        elif len(new_list) < l:
            new_list.extend(random_sample(dummy_list, l - len(new_list)))
        sampled_value = new_list[np.random.randint(0, l, 1)[0]]
        # Ensure the list is exactly 100 elements
        user_movie_lists[user_id] = sampled_value

    # Step 3: Uniformly sample one value from each user's list and print
    sampled_values = []

    for user_id, sampled_value in user_movie_lists.items():
        # sampled_value = np.random.choice(movie_list)
        sampled_values.append((user_id, sampled_value[0], sampled_value[1]))

    with open('movie_paddingSimple_100.txt', 'w') as file:
        for sampled_value in sampled_values:
            print(
                f"User ID: {sampled_value[0]}, Sampled Movie ID: {sampled_value[1]}, Sampled rating {sampled_value[2]}")
            file.write(f'{sampled_value[1]},{sampled_value[2]}\n')
        file.close()


def generateData():
    data = [
        318, 356, 296, 2571, 593, 260, 2959, 480, 527, 4993,
        1196, 110, 1, 589, 50, 1210, 5952, 7153, 1198, 858,
        2858, 47, 1270, 58559, 2028, 608, 79132, 3578, 780, 457,
        2762, 150, 32, 4306, 4226, 364, 1704, 588, 592, 6539
    ] # [1, 50, 150, 260, 296, 318, 356, 457, 527, 589, 592, 593, 1210, 1270, 2571, 2858, 2959, 4993, 7153, 58559, 79132]
    first_column_set = set(data)
    data_dict = defaultdict(list)
    with open('movie_paddingSimple_final.txt', 'r') as file:
        for line in file:
            col1, col2 = line.strip().split(',')
            if int(col1) in first_column_set:
                data_dict[int(col1)].append(float(col2))
        file.close()
    print(data_dict.get(318))
    # print(data_dict.keys(), data_dict.values())

# a = [1, 6, 10, 11, 14, 25, 36, 39, 47, 48, 50, 70, 79, 110, 140, 151, 161, 165, 173, 208, 215, 225, 231, 235, 237, 246, 260, 277, 292, 296, 303, 317, 319, 327, 337, 342, 344, 348, 350, 356, 364, 367, 368, 377, 381, 434, 440, 454, 457, 471, 474, 524, 531, 541, 552, 586, 588, 593, 595, 608, 628, 719, 741, 745, 783, 799, 830, 858, 912, 913, 1036, 1042, 1073, 1092, 1120, 1186, 1188, 1193, 1196, 1197, 1198, 1199, 1200, 1204, 1207, 1208, 1210, 1213, 1215, 1219, 1222, 1225, 1228, 1233, 1242, 1246, 1263, 1274, 1296, 1302, 1347, 1358, 1372, 1377, 1394, 1396, 1566, 1610, 1676, 1717, 1747, 1748, 1777, 1784, 1909, 1917, 1921, 1953, 1962, 1968, 1982, 1997, 2000, 2005, 2019, 2023, 2058, 2078, 2109, 2150, 2161, 2248, 2288, 2321, 2393, 2406, 2455, 2470, 2501, 2539, 2571, 2599, 2699, 2701, 2707, 2746, 2763, 2804, 2826, 2858, 2871, 2881, 2890, 2959, 2987, 3052, 3147, 3160, 3174, 3253, 3421, 3448, 3471, 3535, 3555, 3623, 3702, 3717, 3753, 3755, 3994, 3996, 4014, 4016, 4027, 4148, 4262, 4306, 4571, 4643, 4720, 4776, 4816, 4973, 5299, 5349, 5377, 5378, 5952, 6016, 6373, 6377, 6378, 6537, 6711, 6934, 7153, 7438, 8361, 8368, 8784, 8874, 30707, 31658, 31696, 32587, 33166, 35836, 40815, 44195, 45720, 47099, 47610, 49272, 55247, 55765, 56367, 56782, 58559, 59615, 59784, 60684, 63992, 64614, 66097, 68157, 68954, 69122, 69757, 71535, 79091, 81834, 81845, 81847, 82459, 85414, 88125, 88129, 88140, 89745, 91529, 92259, 94959, 96821, 97938, 98809, 102125, 104841, 106100, 106920, 115149, 115617, 122916, 122922, 148626, 166461, 166528, 168250, 177765, 202439, 207313]
a = [1, 47, 50, 150, 216, 231, 256, 260, 296, 318, 349, 356, 480, 527, 593, 628, 661, 788, 858, 899, 919, 953, 1036, 1073, 1210, 1235, 1617, 1721, 1884, 1917, 1923, 1968, 2020, 2571, 2694, 2871, 2959, 2997, 3555, 4226, 4239, 4306, 4993, 5464, 5952, 7153, 8368, 8464, 30749, 48385, 53125, 56367, 57669, 79132, 79702, 106487]

print(len(a))
# generateData()
# paddingSampling(l=100)
infor()
