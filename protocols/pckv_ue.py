import numpy as np
import math
import random


# pckv ue 的扰动情况
def perturb_ue(v, p1, p2):
    if v > 0:
        p = random.random()
        if p < p1:
            return 0
        elif p <= 0.5:
            return 1
        return 2
    elif v < 0:
        p = random.random()
        if p < p1:
            return 1
        elif p <= 0.5:
            return 0
        return 2


def perturb_ue2(p4):
    p = random.random()
    if p < p4:
        return 0
    elif p < 2 * p4:
        return 1
    return 2


def pckv_UE(data, epsilonls, real_f=0, real_mean=0, paddinglength=1):
    reslist_f = []
    reslist_v = []
    reslist_sum = []
    realsum = sum(data[0])
    n = len(data[0]) + len(data[1])
    if real_f == 0:
        real_f = len(data[0]) / n * paddinglength
        real_mean = sum(data[0]) / len(data[0])
    # print("this is pckv_UE,real_f is", real_f)
    for epsilon in epsilonls:
        e_epsilon = math.e ** epsilon
        mse_f, mse_v, mse_sum = 0, 0, 0
        for _ in range(1):
            sum_average_all = [sum(i) / len(i) for i in data]
            # for dataIndex in range(data_types):
            res_all_x = [0, 0, 0]
            p1 = e_epsilon / (1 + e_epsilon) / 2
            p2 = 1 / (1 + e_epsilon) / 2
            p3 = (e_epsilon + 1) / (e_epsilon + 3)
            p4 = 1 / (e_epsilon + 3)
            # 扰动过程～～～
            for dataIndex in range(1):
                for v in data[0]:
                    round_v = rounding(v)
                    index = perturb_ue(round_v, p1, p2)
                    res_all_x[index] += 1
                # 没用的信息，扰动必要的欸欸欸欸
                for _ in data[1]:
                    index = perturb_ue2(p4)
                    res_all_x[index] += 1
            # 然后对所有的 res_all_x revision
            estimate_f, estimate_v = revise_ue(n, res_all_x[0], res_all_x[1], e_epsilon)

            # estimate_v = 0
            # n1 n2 合并，并且回归求到均值。
            if estimate_f < 0:
                estimate_f = 0
            if estimate_v > 1:
                estimate_v = 1
            elif estimate_v < -1:
                estimate_v = -1
            estimate_sum = estimate_v * estimate_f * paddinglength * n
            # print((n1+n2)/n)
            # estimate_v[index_revise] = (n1 - n2) / n / (1/data_types)
            # print(estimate_v,sum_average_all[0])
            # print("PCKV-UE: \t",estimate_v,estimate_f*paddinglength)
            mse_f += (estimate_f * paddinglength - real_f) ** 2
            mse_v += (estimate_v - real_mean) ** 2
            mse_sum += (estimate_sum - realsum * paddinglength) ** 2
            # index_revise+=1
        reslist_f.append(mse_f / 1)
        reslist_v.append(mse_v / 1)
        reslist_sum.append(mse_sum / 1)
    # print(reslist)
    return reslist_f, reslist_v, reslist_sum


def revise_ue(n, n1, n2, Grr_ep):
    a = 1 / 2
    b = 1 / ((Grr_ep + 1) / 2 + 1)
    p = Grr_ep / (Grr_ep + 1)
    return ((n1 + n2) / n - b) / (a - b), (n1 - n2) * (a - b) / a / (2 * p - 1) / (n1 + n2 - n * b)


def rounding(v):
    pro = (1 + v) / 2
    if random.random() < pro:
        return 1
    else:
        return -1
