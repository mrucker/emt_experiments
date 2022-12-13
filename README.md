# Eigen Memory Trees

Eigen memory trees (EMT) [[1](https://arxiv.org/abs/2210.14077)] are iterative memory models that learn using both supervised and unsupervised information.

# Motivation

EMTs were developed to possess three qualities:
1. iterative growth (i.e., the tree's memory bank grows one memory at a time)
2. iterative learning (i.e., the tree's search function improves over time)
3. sublinear complexity (i.e., the tree's insertion and query complexity is O(log n))

# Experiments

This repository contains 206 experiments evaluating EMT performance on online contextual bandit problems.

There are two experimental settings that are evaluated:
1. Unbounded -- the tree keeps all the memories it is given (run_unbounded.py)
2. Bounded -- the tree must begin pruning memories once it reaches its bound (run_bounded.py)

# Results

After running the experiments the results can be visualized using `/notebooks/plots.ipynb`
