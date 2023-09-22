# Introduction

While the concept of selecting recombination operators based on the phenotype of individuals is not novel (citation needed), extensive research has already explored the use of evolutionary algorithms for training neural networks (citation needed). However, much of this research has primarily concentrated on evolving the network topology rather than addressing the critical issue of selecting the appropriate components. This paper aims to investigate how different types of recombination operators, specifically uniform crossover and whole arithmetic crossover, can significantly impact the performance of evolutionary algorithms.

# Motivation

In traditional neural network training using backpropagation (citation needed), every weight is adjusted based on the entire network's prediction, resulting in interdependencies among all the network weights (citation needed). Recombination operators in the "crossover family" (a terminology concern) function by partitioning the genomes of parent individuals and recombining them to produce offspring. When applied to neural network weights, this process can disrupt the synergy between the weights entirely. Conversely, operators such as whole arithmetic recombination, which do not split the genome, preserve the relationships between weights. Consequently, we anticipate that these operators will outperform traditional crossover operators.

{Probably we can explain formally how they works with some figure}
