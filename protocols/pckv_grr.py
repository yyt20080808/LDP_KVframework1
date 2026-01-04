import math
import numpy as np
import random


# 这种方案不需要任何的分组，假设有两个key,那么用duchi就将其分成6组，pckv——GRR
def pckv_GRR(data, epsilonls, mul_d, real_f=0, real_mean=0, paddinglength=1):
    # print("this is pckv_grr")
    realsum = sum(data[0])
    reslist_f = []
    reslist_v = []
    reslist_sum = []
    n = len(data[0]) + len(data[1])
    if real_f == 0:
        real_f = len(data[0]) / n * paddinglength
        real_mean = sum(data[0]) / len(data[0])
    for epsilon in epsilonls:
        GRR_epsilon = math.e ** epsilon
        mse_f, mse_v, mse_sum = 0, 0, 0
        for times in range(1):
            # sum_average_all = [sum(i) / len(i) for i in data]
            # for dataIndex in range(data_types):
            res_all_x = [0, 0, 0]
            for dataIndex in range(1):
                for v in data[0]:
                    round_v = rounding(v)
                    if round_v > 0:
                        round_v = 0
                    else:
                        round_v = 1
                    index = perturb1(round_v, (mul_d + paddinglength) * 2, GRR_epsilon, paddinglength)
                    if index >= 2:
                        res_all_x[2] += 1
                    else:
                        res_all_x[index] += 1
                for _ in data[1]:
                    round_v = rounding(0)
                    if round_v > 0:
                        round_v = 2
                    else:
                        round_v = 2
                    index = perturb2(round_v, (mul_d + paddinglength) * 2, GRR_epsilon, paddinglength)
                    if index >= 2:
                        res_all_x[2] += 1
                    else:
                        res_all_x[index] += 1
            estimate_f, estimate_v = revise(res_all_x, GRR_epsilon, (mul_d + paddinglength) * 2, n, paddinglength)
            # n1 = res_revise_x[0]
            # n2 = res_revise_x[1]
            # estimate_v = (n1 - n2) / (n1 + n2)
            # estimate_f = n1+n2
            if estimate_v > 1:
                estimate_v = 1
            elif estimate_v < -1:
                estimate_v = -1
            # print((n1+n2)/n)
            # estimate_v[index_revise] = (n1 - n2) / n / (1/data_types)
            # print(estimate_v, sum_average_all[0])
            estimate_f = estimate_f * paddinglength
            if estimate_f > 1:
                estimate_f = 0.9
            elif estimate_f < 0:
                estimate_f = 0
            mse_v += (estimate_v - real_mean) ** 2
            mse_f += (estimate_f - real_f) ** 2
            estimate_sum = estimate_v * estimate_f * n
            mse_sum += (estimate_sum - realsum * paddinglength) ** 2
        reslist_f.append(mse_f / 1)
        reslist_v.append(mse_v / 1)
        reslist_sum.append(mse_sum / 1)
    return reslist_f, reslist_v, reslist_sum


def perturb1(v, all_length, e_epsilon, l):
    p_true = (l * (e_epsilon - 1) + 1) / (l * (e_epsilon - 1) + 2 * all_length)
    if random.random() < p_true:
        return v
    pro = random.random()
    s = 1 / (all_length - 1)
    b = int(pro / s)
    if b == 0:
        if v == 1:
            return 0
        else:
            return 1
    return 2


def perturb2(v, all_length, e_epsilon, l):
    q = 2 / (l * (e_epsilon - 1) + 2 * all_length)
    if random.random() < q:
        pro = random.random()
        if pro < 1/2:
            return 0
        else:
            return 1
    return 2


def revise(input_value, e_epsilon, all_length, n, l):
    p = (l * (e_epsilon - 1) + 1) / (l * (e_epsilon - 1) + 2 * all_length)
    q = 1 / (l * (e_epsilon - 1) + 2 * all_length)
    n1 = (input_value[0] - q * n) / (p - q)
    n2 = (input_value[1] - q * n) / (p - q)
    return (n1 + n2) / n, (n1 - n2) / (n1 + n2)


def rounding(v):
    pro = (1 + v) / 2
    if random.random() < pro:
        return 1
    else:
        return -1
