import numpy as np
import random
from numpy import linalg as LA


def GRR(e_epsilon, input_v, total):
    p_true = e_epsilon / (e_epsilon + total - 1)
    pro = random.random()
    if pro < p_true:
        return input_v
    else:
        v = np.random.randint(0, total - 1)
        if v >= input_v:
            v += 1
        return v


res = []


def noisedP2hist(noised_report, num_GRR):
    a = [0 for i in range(num_GRR)]
    for v in noised_report:
        a[v] += 1
    return np.array(a)


def gen_new_matrix(e_epsilon, number):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    new_matrix = np.full((number, number + 1), q)

    # Set diagonal elements to p
    np.fill_diagonal(new_matrix, p)
    for i in range(number):
        for j in range(number):
            new_matrix[i, j] = new_matrix[i, j]
        new_matrix[i, number] = 1 / (number)

    return new_matrix


def EM_GRR(noised_report, num_GRR, epsilon, length):
    e_epsilon = np.exp(epsilon)
    loglikelihood_threshold = e_epsilon / 1000000
    matrix = gen_new_matrix(e_epsilon, num_GRR)

    weights = np.ones(num_GRR + 1) / num_GRR / 10
    weights[num_GRR] = 0.9
    weights[0:length] = 0
    ns_hist = noisedP2hist(noised_report, num_GRR)
    # sorted_indices = np.argsort(ns_hist)
    ttt = np.sum(ns_hist[0:length])
    for t in range(length):
        ns_hist[t] = ttt / length
    weights_old = np.zeros(num_GRR + 1)
    old_logliklihood = 0
    r = 0
    while LA.norm(weights_old - weights, ord=1) > 1 / 100000 and r < 15000:
        weights_old = np.copy(weights)
        X_condition = np.matmul(matrix, weights_old)
        TMP = matrix.T / X_condition
        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * weights_old
        weights = np.copy(P / sum(P))

        logliklihood = np.inner(ns_hist, np.log(np.matmul(matrix, weights)))
        imporve = logliklihood - old_logliklihood

        r += 1
        if r > 1500 and abs(imporve) < loglikelihood_threshold:
            # print(r)
            break
        # weights[0] = 0
        # if r >= 300 and r % 100 == 0:
        #     for i in range(len(weights)-1):
        #         if weights[i] < 0.02:
        #             weights[i] = 0

        old_logliklihood = logliklihood
    return weights


def gen_matrix(e_epsilon, number):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    # Initialize the matrix with all elements set to q
    matrix = np.full((number, number), q)

    # Set diagonal elements to 1/2
    np.fill_diagonal(matrix, p)
    return matrix


def GRR_revision(noised_reports, num_GRR, epsilon):
    matrix = gen_matrix(np.exp(epsilon), num_GRR)
    ns_hist = noisedP2hist(noised_reports, num_GRR)[0:num_GRR]
    ns_hist = ns_hist / len(noised_reports)

    c = np.matrix(matrix)
    d = np.matmul(c.I, ns_hist)
    d = np.asarray(d)
    return d[0]
