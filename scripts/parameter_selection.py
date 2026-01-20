import numpy as np
import math

from protocols.ours_grr_em  import EM_GRR,GRR
from protocols.ours import sw_perturb,generateMatrix,myEM_smooth_fast_select
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets"
def generate_gaussian(
        bins: int,
        total_count: int = 50000,
        mu: float = 1.0,
        var: float = 0.5,
):
    """

    """
    sigma = np.sqrt(var)

    x_min = mu - 2 * sigma
    x_max = mu + 2 * sigma

    # bin 划分
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian PDF
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)

    # 离散概率
    probabilities = pdf * bin_width
    probabilities /= probabilities.sum()  # 防数值误差

    # 映射到整数 counts
    discrete_counts = np.round(probabilities * total_count).astype(int)

    # 修正 rounding 误差，保证总和严格等于 total_count
    diff = total_count - discrete_counts.sum()
    discrete_counts[np.argmax(discrete_counts)] += diff

    return discrete_counts
def histogram_to_list_random(hist, low=-1.0, high=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    hist = np.asarray(hist, dtype=int)
    a = len(hist)
    bin_edges = np.linspace(low, high, a + 1)

    values = []
    for i, count in enumerate(hist):
        if count > 0:
            samples = np.random.uniform(
                bin_edges[i], bin_edges[i + 1], size=count
            )
            values.append(samples)

    return np.concatenate(values).tolist()

def generate_noised_data(epsilon,a,ifgrr=True,dummy=450000):
    #
    if ifgrr:
        ori = np.array(generate_gaussian(bins=20,total_count=int(500000-dummy)))
    else:
        ori = np.array(generate_gaussian(bins=256,total_count=int(500000-dummy)))
    extend_M = np.zeros(a, dtype=ori.dtype)
    ori = np.concatenate([extend_M, ori])
    number = len(ori)
    ori = np.rint(ori).astype(int)
    noised_data = []
    w = 0
    if ifgrr:
        for i in range(len(ori)):
            for _ in range(ori[i]):
                noised_data.append(GRR(math.e ** epsilon, i, number)) # apply GRR on this extended domain
        random_numbers = [random.randint(0, len(ori)-1) for _ in range(dummy)] # dummy values accross the domain
        noised_data.extend(random_numbers)
        noised_data = np.array(noised_data)
    else:
        data = histogram_to_list_random(ori)
        w,noised_data = sw_perturb(epsilon, data)
        uni_randoms = np.random.uniform(-w / 2, 1 + w / 2, dummy) # dummy values accross the domain
        noised_data =  np.concatenate((noised_data,uni_randoms))
    real_f = sum(ori)/(sum(ori)+dummy)
    return noised_data, number,real_f,ori,w


def calmean(weights,domain):
    weights = weights / sum(weights)
    sums = 0
    for v in range(domain):
        sums += (v+1)*weights[v]
    sums = (sums-domain/2)/domain
    return sums

def grr_selection():
    epsilon = 1
    mse_list_f = []
    mse_list_mean = []
    exper_time = 20
    for v in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        acc_f = 0
        acc_mean = 0
        for i in range(exper_time):
            noised_data2, number, real_f, ori,w = generate_noised_data(epsilon, v,dummy=475000)
            weights = EM_GRR(noised_data2, number, epsilon, v)
            est_mean_ours = calmean(weights[v:20 + v],20)
            real_mean = calmean(ori[v:],20)
            acc_f += (sum(weights[:-1]) - real_f) ** 2
            acc_mean += (est_mean_ours - real_mean) ** 2
        mse_list_f.append(acc_f / exper_time)
        mse_list_mean.append(acc_mean / exper_time)
    radio = [i/20 for i in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]]
    print("parameter",radio)
    print("mse_list_f", mse_list_f)
    print("mse_list_mean", mse_list_mean)

def sw_selection():
    epsilon = 2
    mse_list_f = []
    mse_list_mean = []
    exper_time = 20

    for v in [12, 25, 37, 50, 64, 80, 100, 128, 192, 256]:
        acc_f = 0
        acc_mean = 0
        for i in range(exper_time):
            noised_data2, number, real_f, ori,w = generate_noised_data(epsilon, v,ifgrr=False,dummy=485000)
            Matrix = generateMatrix(epsilon, number)
            noisedData, _ = np.histogram(noised_data2, bins=number, range=(-w / 2, 1 + w / 2))
            weights =  myEM_smooth_fast_select(number, noisedData,Matrix,  10000, 0.0001 * np.exp(epsilon),v)
            est_mean_ours = calmean(weights[v:256 + v], 256)
            real_mean = calmean(ori[v:],256)
            acc_f += (sum(weights[:-1]) - real_f) ** 2
            acc_mean += (est_mean_ours - real_mean) ** 2
        mse_list_f.append(acc_f / exper_time)
        mse_list_mean.append(acc_mean / exper_time)
    radio = [i / 256 for i in [12, 25, 37, 50, 64, 80, 100, 128, 192, 256]]
    print("parameter", radio)
    print("mse_list_f", mse_list_f)
    print("mse_list_mean", mse_list_mean)
if __name__ == "__main__":
    # grr_selection()
    sw_selection()
