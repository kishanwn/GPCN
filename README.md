## Introduction

This repository contains the code and data for the Graph Polynomial Convolution Model. The basic mode is GPCN which consists of convolution by higher-order normalized adjacency matrices with higher-orders of weight matrices. This model has a variation GPCN-LINK (AGPCN) which combined the graph polynomial model with direct learning from the normalized adjacency matrix using adaptive scaling. Further, there are full-adaptive models of AGPCN and AGPCN-LINK which learn the coefficients of higher-order convolutions.    

## Dependencies
- pytoch
- numpy
- scipy

## Data

We used two sources for node-calssification data. The well known datasets Cora, Citeseer,... datasets are from [1]. We also used the recently published non-homopuilous data from [2] and [3]. We converted the non-homogenous datasets from [2] and [3] to a convenient .pt format that is available in the newdata_split folder. 

## Experiments

Use old_data.sh to run the experiment for data from [1]. The new_data.sh contains the experimental setup for the non-homogenous data fron [2] and [3].




[1] Pei et al., 2020] Pei, H., Wei, B., Chang, K. C., Lei, Y., and Yang, B. (2020). Geom-gcn: Geometric graph convolutional networks. In ICLR 2020, ICLR’20.

[2] [Lim et al., 2021b] Lim, D., Li, X., Hohne, F., and Lim, S.-N. (2021b). New benchmarks for learning on non-homophilous graphs. Workshop on Graph Learning Benchmarks, WWW 2021.

[3] [Lim et al., 2021a] Lim, D., Hohne, F. M., Li, X., Huang, S. L., Gupta, V., Bhalerao, O. P., and Lim, S.-N. (2021a). Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. In NeurIPS.
