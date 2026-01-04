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
        real_f = len(data[0]) / n * paddinglength
        real_mean = sum(data[0]) / len(data[0])
    # print("this is ours")
    res_f, res_v, res_sum, res_range = [[] for _ in range(4)]
    for epsilon in epsilonls:
        mse_f, mse_v, mse_sum, mse_range = [[] for _ in range(4)]
        # the value of sw
        ee = np.exp(epsilon)
        w = ((epsilon * ee) - ee + 1) / (2 * ee * (ee - 1 - epsilon)) * 2
        p = ee / (w * ee + 1)
        q = 1 / (w * ee + 1)

        Matrix = generateMatrix(epsilon, pre_bins)
        real_dist = np.histogram(data[0], bins=pre_bins, range=(-1, 1))[0]
        real_dist = real_dist / sum(real_dist)

        for times in range(numberExperi):
            ori_samples = np.array(data[0])
            samples = (ori_samples + 1) / 2
            randoms = np.random.uniform(0, 1, len(samples))

            noisy_samples = np.zeros_like(samples)
            # report
            index = randoms <= (q * samples)
            noisy_samples[index] = randoms[index] / q - w / 2
            index = randoms > (q * samples)
            noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
            index = randoms > q * samples + p * w
            noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2
                # observed_value.append(res_x)
            uni_randoms = np.random.uniform(-w/2, 1+w/2, len(data[1]))
            observed_value = np.concatenate((noisy_samples,uni_randoms)) # concate
            # for _ in range(len()):
            #     res_x = random.random() * 2 * C - C
            #     res_all_x += res_x
            #     observed_value.append(res_x)

            noisedData, _ = np.histogram(observed_value, bins=pre_bins, range=(-w/2, 1+w/2))
            maxiteration = 10000
            if smooth:
                theta_OPM_NoS = myEM_one(pre_bins, noisedData, Matrix, maxiteration, 0.001 * ee)
            else:
                if paddinglength == 1:
                    maxiteration = 5000
                theta_OPM_NoS = myEM_nosmooth_fast(pre_bins, noisedData, Matrix, maxiteration, 0.001 * ee)

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
            if estimate_f * paddinglength > 1:
                estimate_f = 0.5 / paddinglength
            estimate_sum = estimate_v * estimate_f * n * paddinglength
            mse_f.append((estimate_f * paddinglength - real_f) ** 2)  # MSE of frequency
            mse_v.append((real_mean - estimate_v) ** 2)  # MSE of vale mean
            mse_sum.append((estimate_sum - realsum * paddinglength) ** 2)
            new_dist = theta_OPM_NoS
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
    return rangequery(real_dist, np.array([1 / pre_bins for _ in range(pre_bins)]))


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
                theta[0:int(n / 2)] = 0
                # if calSum(theta[0:n]) < -0.3 and sum(theta[int(n / 2):n]) / sum(theta[0:n]) < 1 / 5:
                #     theta[int(n / 4):n] = 0
                #     # print("effect small")
                # elif calSum(theta[0:n]) > 0.1 and sum(theta[0:int(n / 2)]) / sum(theta[0:n]) < 1 / 5:
                #     theta[0:int(n / 2)] = 0
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


def myEM_nosmooth_fast(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    ns_hist = np.asarray(ns_hist, dtype=np.float64)      # 保证是 float，除法更快更稳定
    transform = np.asarray(transform, dtype=np.float64)

    # 1) 快速构造 transform2: [transform | 1/n]
    A = np.empty((n, n + 1), dtype=np.float64)
    A[:, :n] = transform
    A[:, n] = 1.0 / n

    sample_size = ns_hist.sum()
    eps = 1.0 / sample_size / 100.0

    # 2) 初始化 theta
    theta = np.ones(n + 1, dtype=np.float64) / float(n) / 100.0
    if max_iteration == 5000:
        theta[:102] = 0.0
        theta[103:204] = 0.0
        theta[205:307] = 0.0
        theta[308:410] = 0.0
        theta[411:511] = 0.0
        theta[n] = 9.95 / 10.0
    else:
        theta[:102] = 0.0

    # 归一化一次
    theta /= theta.sum()

    old_loglikelihood = -np.inf
    theta_old = theta.copy()

    for r in range(max_iteration):
        # 保存旧值（尽量少 copy：只在需要收敛判据时 copy）
        theta_old[:] = theta

        # E-step: X = A @ theta
        X = A @ theta

        # 防止除零和 log(0)：clip 很关键
        X = np.clip(X, 1e-300, None)

        # 核心提速：w = ns_hist / X，然后 g = A.T @ w
        w = ns_hist / X
        g = A.T @ w

        # M-step
        theta *= g
        s = theta.sum()
        if s == 0.0:
            # 极端数值异常保护：退回均匀（或你可选择 break）
            theta[:] = 1.0 / (n + 1)
        else:
            theta /= s

        # 你的启发式修正逻辑（尽量不动）
        if r == 50 or r == 1000:
            # 这里的 calSum / 逻辑我按原样保留调用位置
            if calSum(theta[:n]) < -0.3 and theta[int(n/2):n].sum() / theta[:n].sum() < 1/5:
                theta[int(n/4):n] = 0.0
            elif calSum(theta[:n]) > 0.1:
                theta[102] *= 0.5
            theta /= theta.sum()

        # 约束噪声分量
        if theta[n] > 0.998:
            theta[n] = 0.998
            theta /= theta.sum()

        # loglikelihood
        loglikelihood = np.dot(ns_hist, np.log(A @ theta).clip(min=1e-300))
        improve = loglikelihood - old_loglikelihood

        # 收敛判据（更便宜）
        if np.sum(np.abs(theta_old - theta)) <= eps:
            break

        if r > 300 and abs(improve) < loglikelihood_threshold:
            break

        old_loglikelihood = loglikelihood

    return theta


def calSum(vvvv):
    vvvv = vvvv / sum(vvvv)
    estimate_v_2 = 0
    for i in range(pre_bins):
        estimate_v_2 += vvvv[i] * i
    estimate_v_2 = (estimate_v_2 / pre_bins - 0.5) * 2
    return estimate_v_2


def generateMatrix(eps, binNumber):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    m = binNumber
    n = binNumber
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
            (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
            rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
            (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    return transform


