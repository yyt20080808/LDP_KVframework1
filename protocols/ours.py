import matplotlib.pyplot as plt
import numpy as np
import math
import random
from numpy import linalg as LA

pre_bins = 512


def OURS_UE(data, epsilonls, real_f=0, real_mean=0, paddinglength=1, smooth=True):
    numberExperi = 1
    realsum = sum(data[0])
    n = len(data[0]) + len(data[1])
    if real_f == 0:
        real_f = len(data[0]) / n *paddinglength
        real_mean = sum(data[0]) / len(data[0])
    # print("this is ours")
    res_f, res_v, res_sum, res_range = [[] for _ in range(4)]
    for epsilon in epsilonls:
        mse_f, mse_v, mse_sum, mse_range = [[] for _ in range(4)]
        # the value of PM
        e_epsilon = math.e ** epsilon
        e_epsilon_sqrt = math.e ** (epsilon / 2)
        B = (e_epsilon_sqrt + 9) / (10 * e_epsilon_sqrt - 10)
        k = B * e_epsilon / (B * e_epsilon - 1 - B)
        C = k + B
        p = 1 / (2 * B * e_epsilon + 2 * k)
        pr = 2 * B * e_epsilon * p

        Matrix = generateMatrix(epsilon, pre_bins)
        real_dist = np.histogram(data[0], bins=pre_bins,range = (-1,1))[0]
        real_dist = real_dist / sum(real_dist)

        for times in range(numberExperi):
            observed_value = []
            res_all_x = 0
            for j in data[0]:
                res_x = OPMnoise(j, k, B, pr, C)
                res_all_x += res_x
                observed_value.append(res_x)
            for _ in range(len(data[1])):
                res_x = random.random() * 2 * C - C
                res_all_x += res_x
                observed_value.append(res_x)

            noisedData, _ = np.histogram(observed_value, bins=pre_bins, range=(-C, C))
            maxiteration = 10000
            if smooth:
                theta_OPM_NoS = myEM_one(pre_bins, noisedData, Matrix, maxiteration, 0.001 * e_epsilon)
            else:
                if paddinglength==1:
                    maxiteration = 5000
                theta_OPM_NoS = myEM_nosmooth(pre_bins, noisedData, Matrix, maxiteration, 0.001 * e_epsilon)

            estimate_f = 1 - theta_OPM_NoS[-1]
            theta_OPM_NoS = theta_OPM_NoS[0:-1]
            theta_OPM_NoS = theta_OPM_NoS / sum(theta_OPM_NoS)
            # show the distribution
            # plt.plot([i for i in range(pre_bins)], theta_OPM_NoS)
            # plt.plot([i for i in range(pre_bins)], real_dist)
            # plt.show()
            # print("sum",sum(theta_OPM_NoS))
            estimate_v = 0
            for i in range(pre_bins):
                estimate_v += theta_OPM_NoS[i] * i
            estimate_v = (estimate_v / pre_bins - 0.5) * 2
            # print("EM:", real_mean, estimate_v, "estimated f:", estimate_f*paddinglength, real_f)
            if estimate_f*paddinglength > 1:
                estimate_f = 0.5/paddinglength
            estimate_sum = estimate_v * estimate_f * n *paddinglength
            mse_f.append((estimate_f * paddinglength - real_f) ** 2)  # MSE of frequency
            mse_v.append((real_mean - estimate_v) ** 2)  # MSE of vale mean
            mse_sum.append((estimate_sum - realsum*paddinglength) ** 2)
            new_dist =  theta_OPM_NoS
            mse_range.append(rangequery(real_dist, new_dist))
        res_f.append(sum(mse_f) / numberExperi)
        res_v.append(sum(mse_v) / numberExperi)
        res_sum.append(sum(mse_sum) / numberExperi)
        res_range.append(sum(mse_range) / numberExperi)
        # res_var1.append(np.var(var1))
        # res_var2.append(np.var(var2))

    # print(reslist)
    # print(res1)
    # print(res2)
    return res_f, res_v, res_sum, res_range


def getrangemse(a):
    real_dist = np.histogram(a, bins=pre_bins)[0]
    real_dist = real_dist / sum(real_dist)
    return rangequery(real_dist, np.array([1/pre_bins for _ in range(pre_bins)]))

def rangequery(a, b):
    cum_a = np.cumsum(a, dtype=np.float64)
    cum_b = np.cumsum(b, dtype=np.float64)

    total_squared_error = 0.0
    for _ in range(30):
        # ensure 0 <= start <= end < n
        r1 = random.randint(0, pre_bins - 1)
        r2 = random.randint(0, pre_bins - 1)
        start, end = min(r1, r2), max(r1, r2)

        # 使用累积和数组 O(1) 计算范围内的计数值
        # sum(arr[start:end+1]) 等价于 cum_arr[end] - cum_arr[start-1]
        count_a = cum_a[end] - (cum_a[start - 1] if start > 0 else 0)
        count_b = cum_b[end] - (cum_b[start - 1] if start > 0 else 0)

        # 计算该次查询的平方误差 (SE)
        squared_error = (count_a - count_b) ** 2
        total_squared_error += squared_error

    # --- 4. 计算平均平方误差 (MSE) ---
    mean_squared_error = total_squared_error / 30

    return mean_squared_error


def myEM_one(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    smoothing_factor = 2
    binomial_tmp = [1, 2, 1]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T
    transform2 = np.ones((n, n + 1))
    for i in range(n):
        for j in range(n):
            transform2[i, j] = transform[i, j]
        transform2[i, n] = 1 / n

    sample_size = sum(ns_hist)
    r = 0
    theta_final = (np.ones(n + 1)) / (float(n))
    loglikelihood_final = -10000000
    for jjj in range(1):

        theta = (np.ones(n + 1)) / (float(n)) / 10
        theta[0:200] = 0.000000
        # theta[0:int(pre_bins / 9)] = 0.0000001
        # theta[900:] = 0.0000001
        theta[n] = 9 / 10

        theta_old = np.zeros(n + 1)
        old_logliklihood = 0
        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size / 100 and r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform2, theta_old)
            TMP = transform2.T / X_condition
            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))
            theta[0:n] = np.matmul(smoothing_matrix, theta[0:n])
            if r == 200 or r == 1000:
                if calSum(theta[0:n]) < -0.3 and sum(theta[int(n / 2):n]) / sum(theta[0:n]) < 1 / 5:
                    theta[int(n / 4):n] = 0
                    # print("effect small")
                elif calSum(theta[0:n]) > 0.1 and sum(theta[0:int(n / 2)]) / sum(theta[0:n]) < 1 / 5:
                    theta[0:int(n / 2)] = 0
                    # print("effect big")
            theta = theta / sum(theta)

            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform2, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 100 and abs(imporve) < loglikelihood_threshold:
                break

            old_logliklihood = logliklihood

            r += 1
        # print("r=", r, "noise components:", theta[n], old_logliklihood)
        # plt.plot([i for i in range(pre_bins)], theta[0:-1])
        # plt.show()
        # print("sum", sum(theta))
        if old_logliklihood > loglikelihood_final:
            theta_final = np.copy(theta)
            loglikelihood_final = old_logliklihood
    return theta_final
def myEM_nosmooth(n, ns_hist, transform, max_iteration, loglikelihood_threshold):

    transform2 = np.ones((n, n + 1))
    for i in range(n):
        for j in range(n):
            transform2[i, j] = transform[i, j]
        transform2[i, n] = 1 / n

    sample_size = sum(ns_hist)
    r = 0
    theta = (np.ones(n + 1)) / (float(n)) / 100
    if max_iteration==5000:
        theta[0:102] = 0.000000
        theta[103:204] = 0.000000
        theta[205:307] = 0.000000
        theta[308:410] = 0.000000
        theta[411:511] = 0.000000
        theta[n] = 9.95 / 10
    else:
        theta[0:102] = 0.000000
    theta_old = np.zeros(n + 1)
    old_logliklihood = 0
    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size / 100 and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform2, theta_old)
        TMP = transform2.T / X_condition
        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))
        # theta[0:n] = np.matmul(smoothing_matrix, theta[0:n])
        if r == 50 or r == 1000:
            # if calSum(theta[0:n]) < -0.3 and sum(theta[int(n / 2):n]) / sum(theta[0:n]) < 1 / 5:
            #     theta[int(n / 4):n] = 0
            #     print("effect small")
            # elif calSum(theta[0:n]) > 0.1 and sum(theta[0:int(n / 2)]) / sum(theta[0:n]) < 1 / 5:
            # theta[0:int(n / 4)] = 0
            #     print("effect big")
            theta[102] = theta[102]/2
        theta = theta / sum(theta)
        if theta[512] > 0.998:
            theta[512] = 0.998
        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform2, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 300 and abs(imporve) < loglikelihood_threshold:
            break

        old_logliklihood = logliklihood

        r += 1
    # print("r=", r, "noise components:", theta[n], old_logliklihood)
        # plt.plot([i for i in range(pre_bins)], theta[0:-1])
        # plt.show()
        # # print("sum", sum(theta))

    return theta



def calSum(vvvv):
    vvvv = vvvv / sum(vvvv)
    estimate_v_2 = 0
    for i in range(pre_bins):
        estimate_v_2 += vvvv[i] * i
    estimate_v_2 = (estimate_v_2 / pre_bins - 0.5) * 2
    return estimate_v_2




def generateMatrix(eps, binNumber):
    e_epsilon = math.e ** eps
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = (e_epsilon_sqrt + 9) / (10 * e_epsilon_sqrt - 10)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)
    # avg_q = 1 / binNumber
    m = binNumber
    n = binNumber
    m_cell = (2 * C) / m
    transform = np.ones((binNumber, binNumber)) * q * m_cell
    q_value = transform[0, 0]
    p_value = q_value * e_epsilon
    pro_pass = 0
    left_index = 0
    right_index = 0
    # 计算 一列中多少个
    for i in range(0, n):
        if i == 0:
            a = int(B / C * n)
            reseverytime = 1 - a * p_value - (n - a) * q_value
            pro_pass = ((n - a - 1) * (p_value - q_value) + (p_value - q_value - reseverytime)) / (n - 1)
            transform[0:a, i] = p_value
            transform[a, i] = q_value + reseverytime
            right_index = a
            left_index = 0
        else:
            temp_right = transform[left_index, i - 1] - pro_pass
            if temp_right >= q_value:
                transform[left_index, i] = transform[left_index, i - 1] - pro_pass  # 左边减去 1
                if transform[right_index, i - 1] + pro_pass < p_value:
                    transform[right_index, i] = transform[right_index, i - 1] + pro_pass
                else:  # 说明要 right_index+1
                    overflow = transform[right_index, i - 1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index >= n - 1 and overflow < 1e-5:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index += 1
                    transform[right_index, i] = q_value + overflow
            else:
                overflow = pro_pass - (transform[left_index, i - 1] - q_value)
                transform[left_index, i] = q_value
                left_index += 1
                transform[left_index, i] = p_value - overflow
                if transform[right_index, i - 1] + pro_pass < p_value:
                    transform[right_index, i] = transform[right_index, i - 1] + pro_pass
                else:  # 说明要 right_index+1
                    overflow = transform[right_index, i - 1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index == n - 1 and overflow < 1e-4:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index += 1
                    transform[right_index, i] = q_value + overflow
            for jjj in range(left_index + 1, right_index):
                transform[jjj, i] = p_value
    return transform



def OPMnoise(x, k, B, pr, C):
    lt = k * x - B
    rt = lt + 2 * B
    # 回答很好的条件
    if random.random() <= pr:
        res = random.random() * 2 * B + lt
    else:
        temp = random.random()
        ppp = (lt + C) / (2 * k)
        if ppp > temp:
            res = temp * (2 * k) - C
        else:
            res = rt + (temp - ppp) * (2 * k)
    return res
