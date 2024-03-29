# Eigen Memory Trees
The eigen memory trees (EMT) paper can be found at [[1](https://arxiv.org/abs/2210.14077)].

# Motivation

EMTs were developed to possess three qualities:
1. iterative growth (i.e., the tree's memory bank grows one memory at a time)
2. iterative learning (i.e., the tree's search function improves one example at a time)
3. sublinear complexity (i.e., the tree's insertion and query complexity is O(log n))

# Experiments

This repository contains two experiments using 206 datasets to evaluate EMT on contextual bandit problems.

There are two experiments provided for EMT:
1. Unbounded -- the tree keeps all the memories it is given (`python run_unbounded.py`)
2. Bounded -- the tree must begin pruning memories once it reaches its bound (`python run_bounded.py`)

# Results

After running the experiments the results can be visualized using `/notebooks/plots.ipynb`

# Dependencies

An `environment.yml` file is provided to create a conda environment for the experiments.
