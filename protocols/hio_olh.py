import numpy as np
import random
import xxhash

def hio(data, epsilonls,real_f = 0, real_mean = 0,paddinglength = 1):
    # print("this is me_olh")
    # print("this is hio")
    realsum = sum(data[0])
    reslist_f = []
    reslist_v = []
    reslist_sum = []
    if real_f==0:
        real_f = len(data[0]) / (len(data[0]) + len(data[1])) *paddinglength
        real_mean = sum(data[0])/len(data[0])

    for epsilon in epsilonls:
        mse_f, mse_v, mse_sum = 0, 0,0
        for times in range(1):
            n = 0
            for users in data:
                n += len(users)
            data_types = len(data)
            pureData = [0,0,0] # get the count of each input for estimate
            res_all_x = [0, 0, 0]
            for v in data[0]:
                round_v = rounding(v)
                if round_v > 0:
                    pureData[1]+=1
                else:
                    pureData[0]+=1
            pureData[2] = len(data[1])
            g = max(3,int(np.exp(epsilon)) + 1)
            noisedReport = lh_perturb(pureData, g, 1/2)
            freq =  lh_aggregate(noisedReport,3,g, 1/2,1/g)

            # index_revise = 0

            n1 = freq[1]
            n2 = freq[0]
            if  n1==n2:
                estimate_v = 0
            else:
                estimate_v = (n1 - n2) / (n1 + n2)
            estimate_f = (n1+n2)/n * paddinglength
            if estimate_f > 1:
                estimate_f = 0.9
            if estimate_f < 0:
                estimate_f = 0
            if estimate_v > 1:
                estimate_v = 1
            elif estimate_v < -1:
                estimate_v = -1
            # print((n1+n2)/n)
            # estimate_v[index_revise] = (n1 - n2) / n / (1/data_types)
            mse_v += (estimate_v - real_mean) ** 2
            mse_f += (estimate_f - real_f) ** 2
            estimate_sum = estimate_v * estimate_f * n
            mse_sum += (estimate_sum - realsum*paddinglength) ** 2
            # print("ME_OLH: \t", estimate_v, sum_average_all[0],(estimate_v - sum_average_all[0]) ** 2)
            # index_revise+=1
        reslist_f.append(mse_f / 1)
        reslist_v.append(mse_v / 1)
        reslist_sum.append(mse_sum / 1)
    # print(reslist)
    return reslist_f, reslist_v, reslist_sum

def lh_perturb(real_dist, g, p):
    n = sum(real_dist)
    noisy_samples = np.zeros(n, dtype=object)
    samples_one = np.random.random_sample(n)
    seeds = np.random.randint(0, n, n)

    counter = 0
    for k, v in enumerate(real_dist):
        for _ in range(v):
            y = x = xxhash.xxh32(str(int(k)), seed=seeds[counter]).intdigest() % g

            if samples_one[counter] > p:
                y = np.random.randint(0, g - 1)
                if y >= x:
                    y += 1
            noisy_samples[counter] = tuple([y, seeds[counter]])
            counter += 1
    return noisy_samples
def lh_aggregate(noisy_samples, domain, g, p, q):
    n = len(noisy_samples)
    est = np.zeros(domain, dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                est[v] += 1

    a = 1.0 / (p - q)
    b = n * q / (p - q)
    est = a * est - b

    return est
def rounding(v):
    pro = (1 + v) / 2
    if random.random() < pro:
        return 1
    else:
        return -1

# if __name__ == "__main__":
#     data = generate_Data(0.03)
#     mse_MEOLH,var_MEOLH = MEOLH(data,[1,2,3,4])
#     print(mse_MEOLH,var_MEOLH)