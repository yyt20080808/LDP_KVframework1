import random
import math
import matplotlib.pyplot as plt
import numpy as np

b = 0.4
def twodimensionalSW(m, n, data1, data2, epsilon, numbers):
    b2 = b / (1 - b)  # b = b2(1+b2)
    ee = np.exp(epsilon)
    q_value = 1 / (b * b2 / (n / m + b2) * ee + 1 - b * b2 / (n / m + b2))
    p_value_high = q_value * ee * b * b2 / (n / m + b2)
    # print(p_value_high)
    observed_data_X = []
    observed_data_Y = []
    for (i, j) in zip(data1, data2):
        if random.random() < p_value_high:  #
            a1 = b2 * random.random() + i - b2 / 2
            a2 = b2 * random.random() + j * n / m - b2 / 2
            observed_data_X.append(a1)
            observed_data_Y.append(a2)
        else:
            while (True):
                a1 = random.random() * (1 + b2) - b2 / 2
                a2 = random.random() * (n / m + b2) - b2 / 2
                if math.fabs(a1 - i) < b2 / 2 and math.fabs(a2 - j) < b2 / 2:
                    pass
                # elif which == "newYork" and a1 / (1 + b2) > 0.8 and a2 / (n / m + b2) < 0.5:
                #     pass
                else:
                    observed_data_X.append(a1)
                    observed_data_Y.append(a2)
                    break
    observed_data_X.extend([random.random() * (1 + b2) - b2 / 2 for _ in range(numbers)])  # add uniform noise
    observed_data_Y.extend([random.random() * (1 + b2) - b2 / 2 for _ in range(numbers)])  # add uniform noise
    sum_all = len(observed_data_Y)
    histogram_ori, xedges, yedges = np.histogram2d(data1, data2, bins=(m, n), range=[[0, 1], [0, 1]])
    histogram_ori.astype(int)
    histogram, xedges, yedges = np.histogram2d(observed_data_Y, observed_data_X, bins=(m, n),
                                               range=[[-b2 / 2, 1 + b2 / 2], [-b2 / 2, n / m + b2 / 2]])

    # plt.imshow(histogram, interpolation='nearest', origin='lower',
    #            extent=[0, n, 0, m])
    # plt.grid(True)
    # plt.colorbar()
    # plt.show()
    # print("histogram:", sum(sum(histogram)))
    transform2 = generateMatrix(epsilon, m, n)
    # mytheta = EMS(m,n,histogram.flatten(),transform2)
    mytheta = em_fast(m, n, histogram.flatten(), transform2)
    newTheta = []
    for i in mytheta:
        if i < 1e-5:
            newTheta.append(0)
        else:
            newTheta.append(int(i * sum_all))
    newTheta = np.array(newTheta)
    # print(sum(newTheta))
    temp1 = newTheta[-1]
    # print("old:", temp1)
    newTheta = newTheta[0:-1]
    newTheta.resize((m, n))

    # plt.imshow(newTheta, cmap=plt.cm.bone, vmax=5000, vmin=00, interpolation='nearest', origin='lower',
    #            extent=[0, m, 0, n])
    #
    # plt.grid(True)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # # plt.savefig("ori222222.png")
    # plt.show()

    return newTheta,1-temp1/sum_all


def get_newTransform(transform, m, n):
    newTransform = np.ones((m * n + 1, m * n))
    for i in range(m):
        for j in range(n):
            newTransform[i * n + j, :] = transform[i, j, :]
        newTransform[m * n, :] = 1 / (m * n)
    return newTransform


def em_fast(m, n, ns_hist, transform2, max_iter=8000, tol=1e-3, eps=1e-12):
    # theta 初始化（保持你原来的逻辑）
    theta = np.ones(m * n + 1, dtype=np.float64) / float(m * n)
    theta[m * n] = 0.7
    theta[0] = 0
    theta[m * n - 1] = 0
    theta[m - 2:m] = 0
    theta[m * n - m - 2:m * n - m] = 0

    # 构造 transform，并明确保存 T 与 TT
    T = get_newTransform(transform2, m, n)      # shape: (obs, params) or (params, obs) 看你实现
    # 你原来做了 transform = transform.T，所以这里直接固定成 T 为你迭代里用的那个
    T = T.T                                     # 现在假设 T.shape = (obs_dim, param_dim)
    TT = T.T                                    # param_dim x obs_dim（只做一次）

    old_ll = -np.inf

    for r in range(max_iter):
        # E step: X = T @ theta
        X = T @ theta

        # 避免除零
        X = np.maximum(X, eps)

        # ratio = ns_hist / X   (obs_dim,)
        ratio = ns_hist / X

        # back = TT @ ratio     (param_dim,)
        back = TT @ ratio

        # M step: theta <- theta * back; normalize
        theta *= back
        s = theta.sum()
        if s <= 0 or not np.isfinite(s):
            raise FloatingPointError("theta became invalid (sum<=0 or non-finite).")
        theta /= s

        # log-likelihood
        # ll = <ns_hist, log(T @ theta)>
        X2 = T @ theta
        X2 = np.maximum(X2, eps)
        ll = np.dot(ns_hist, np.log(X2))

        improve = ll - old_ll
        if r > 1 and abs(improve) < tol:
            print("stop when", improve, "at iter", r)
            break
        old_ll = ll

    return theta

def generateMatrix(epsilon, m, n):
    b2 = b / (1 - b)
    d = m * n
    ee = np.exp(epsilon)
    q_value = 1 / (b * b2 / (n / m + b2) * ee + 1 - b * b2 / (n / m + b2)) / d
    p_value = q_value * ee
    transform = np.ones((m, n, d)) * q_value
    for i in range(m):
        for j in range(n):
            left = (j + 0.5) * (1 - b2 / (b2 + n / m))
            right = (j + 0.5) * (1 - b2 / (b2 + n / m)) + n * b2 / (b2 + n / m)
            top = (i + 0.5) * (1 - b)
            boltom = (i + 0.5) * (1 - b) + m * b
            # 首先赋值左上角。
            try:
                left_l = int(left)
                top_l = int(top)
                right_r = int(right)
                boltom_r = int(boltom)
                porportion1 = 1 - (left - left_l)
                porportion2 = 1 - (top - top_l)
                porportion3 = right - right_r
                porportion4 = boltom - boltom_r

                transform[i, j, top_l * n + left_l] = porportion1 * porportion2 * p_value + (
                        1 - porportion1 * porportion2) * q_value

                transform[i, j, top_l * n + right_r] = porportion3 * porportion2 * p_value + (
                        1 - porportion3 * porportion2) * q_value

                # 左下角的那个点，右下角的点
                transform[i, j, (boltom_r) * n + left_l] = porportion1 * porportion4 * p_value + (
                        1 - porportion1 * porportion4) * q_value
                transform[i, j, (boltom_r) * n + right_r] = porportion3 * porportion4 * p_value + (
                        1 - porportion3 * porportion4) * q_value
                # 左边的那一列
                for index in range(left_l + 1, right_r):
                    transform[i, j, top_l * n + index] = porportion2 * p_value + (1 - porportion2) * q_value
                    transform[i, j, (boltom_r) * n + index] = porportion4 * p_value + (1 - porportion4) * q_value
                for index in range(top_l + 1, boltom_r):
                    transform[i, j, index * n + left_l] = porportion1 * p_value + (1 - porportion1) * q_value
                    transform[i, j, index * n + right_r] = porportion3 * p_value + (1 - porportion3) * q_value
                # 将中间的都设置为 p
                for index1 in range(top_l + 1, boltom_r):
                    for index2 in range(left_l + 1, right_r):
                        transform[i, j, index1 * n + index2] = p_value
            except:
                print("")

    return transform
