# PrivECE: A Framework for Enhanced Conditional Estimation on Key-Value Data with Local Privacy Guarantees
This repository provides the implementation of PrivECE, a novel framework for key-value data collection and analysis under pure Local Differential Privacy (LDP) guarantees.

PrivECE focuses on accurate frequency estimation on keys and conditional estimation on values, and proposes an enhanced perturbation and aggregation framework compared with state-of-the-art (SOTA) LDP protocols.

## Dependencies
To run the experiments in this repo, you may need to use python3.8 or later version, you need to install package `numpy`, `xxhash`,`matplotlib`. 

## Outline
This includes the implementations for estimating LDP noised reports generate from SOTA LDP protocols.
- ` [PCKV](https://www.usenix.org/system/files/sec20-gu.pdf): Existing key-value protocols, for frequency estimation on key, and conditional mean estimation on value.

Some FOs code is based on the implementation by [Wang](https://github.com/vvv214/LDP_Protocols) and [Maddock](https://github.com/Samuel-Maddock/pure-LDP/blob/master/README.md). And the estimating methods includes unbiased estimation, post-processing method([Basecut, Normsub](https://github.com/vvv214/LDP_Protocols/tree/master/post-process),[IIW](https://github.com/SEUNICK/LDP)), EM-based MLE and our MR.

## File structure
- `./datasets`: the source directory of all the datasets(e.g., Movie, Taxi), additionally, including the scripts of preparing padding and sampling steps.
  `./datasets/amazonshopping` includes raw datasets and preparing padding and sampling with l=3
- `./protocols`: the source directory of all the mechanisms, models we have experimented with.
  - `./protocols/ours.y` implements our main method, the LVP perturbing method with EM aggregation implements our main method, the LVP pe
  - `./protocols/ours_two_d` implements our main method, the LVP perturbing method where value is two-dimension data, with EM aggregation framework for Key-Value data analysis.
  - `./protocols/hio_olh`  SOTA LDP KV protocols, we implement its EM-based MLE and unbiased estimation (PM).
  - `./protocols/pckv_grr & pckv_oue`  SOTA LDP KV protocols, we implement with fixed $\ell$.
  - `./protocols/privkvm & privkvm_star` SOTA LDP KV protocols, follows the implementation on conditional histogram task on TDSC' 2023.
- `./scripts` is the directory of Python scripts to run experiments on different datasets.
  - `./scripts/synthetic.py` runs the KV results on Power-Law and Gaussian datasets, comparing our results with other SOTA.
  - `./scripts/movie.py` runs the KV results on Movie-ratings datasets, comparing our results with other SOTA.
  - `./scripts/amazon.py` runs the KV results on amazon product-ratings datasets, comparing our results with other SOTA.
  - `./scripts/taxi.py` runs the KV results on newyork Taxi datasets, comparing our results with other OUE.
 
 
## Running
From the top-level directory, you can run :
```
python3 -m scripts.synthetic
```
You can replace synthetic with other datasets (e.g., movie, amazon, taxi).
## Basic Usage
This is an example of running synthetic.py
```python
XXXXXXXX # the functions in this file above are reading datasets 

# Using  for frequency estimation

if __name__ == "__main__":
    # all_list = generate_Data( "gaussian")   # we provide two systhetic datasets as in our paper
    all_list = generate_Data("power-law")
    padding_l = 6                  # pre fixed padding and sampling length.
    number = 1000000               # number of users
    epslions = [2]                 # epsilon 

    m = len(epslions) 
    n = 10                        #

```
And the output is
```
Information of Power-law distribution:
Number of keys reported(Including dummy):105
Real Reports on each key [89362, 51214, 35541, 26569, 21004, 17584, 14956, 12963, 11732, 10240, 9348, 8378, 7761, 7108, 6645, 6187, 5817, 5447, 5058, 4680, 4591, 4344, 4163, 3962, 3793, 3652, 3451, 3343, 3202, 3136, 3072, 2977, 2839, 2790, 2662, 2574, 2579, 2350, 2363, 2289, 2221, 2190, 2111, 2065, 2031, 1961, 1904, 1888, 1888, 1811, 1712, 1702, 1630, 1604, 1684, 1586, 1666, 1542, 1560, 1528, 1422, 1434, 1452, 1409, 1344, 1329, 1312, 1262, 1264, 1278, 1210, 1161, 1201, 1211, 1128, 1144, 1040, 1156, 1025, 1045, 1034, 1054, 1017, 996, 998, 937, 994, 946, 914, 949, 923, 910, 879, 917, 872, 909, 841, 870, 841, 792, 141297, 134646, 123392, 98361, 874]

MSE of frequency:
res_ME_grr =  [0.13237081]
res_ME_OLH =  [0.00044908]
PCKV_UE =  [0.000259038]
ours_f =  [0.00015734]

MSE of values mean:
res_ME_grr =  [0.01421951]
res_ME_OLH =  [0.00041524]
res_PCKV_UE =  [2.39244e-04]
OURS_UE =  [0.00016965]

MSE of values sum:
res_ME_grr =  [1.58570417e+11]
res_ME_OLH =  [1.15789905e+08]
res_PCKV_UE =  [45113746.48163419]
OURS_UE =  [1037042.53960399]
```

## Outputs


\
## Acknowledgements
Parts of the implementation are based on publicly available code from prior work. We sincerely thank the authors for releasing their implementations.
## Todo
Provide scripts for reproducing all figures in the paper
