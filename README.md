# DSD-PRO

PyTorch implementation of **UGC-VQA based on decomposition (dual-stream decompostion, DSD) and recomposition (progressively residual aggregation, PRO)**

A new version v2.0 has been updated. We enable the normalization.

## Overview

[TODO]

## Usage

[TODO]

## Performance

NOTE: We **DO NOT** use nonlinear regression for both PLCC and RMSE for simplicity. 
The median performance is adopted through 30 repetitions.

### Environment
Different environment may induce possible fluctuation of performance.

```
Python 3.6.5
PyTorch 1.1.0
Numpy 1.19.5
Scipy 1.1.0
```


### Intra-dataset
It follows a standard 60%/20%/20% for training/validation/testing within each database. 

|  DB  | KoNViD | LIVE-VQC | YouTube-UGC |
| :--: | :----: | :------: | :---------: |
| SRCC |   0    |    0     |      0      |
| KRCC |   0    |    0     |      0      |
| PLCC |   0    |    0     |      0      |
| RMSE |   0    |    0     |      0      |


### Inter-dataset
The model uses 80%/20 for training/validation in one database, and is directly tested on the others. As we DO NOT utilize nonlinear regression, PLCC and RMSE in the following table seem less appropriate.

| Trained on KoNViD | LIVE-VQC | YouTube-UGC |
| :---------------: | :------: | :---------: |
|       SRCC        |    0     |      0      |
|       KRCC        |    0     |      0      |
|       PLCC        |    0     |      0      |
|       RMSE        |    0     |      0      |

| Trained on LIVE-VQC | KoNViD | YouTube-UGC |
| :-----------------: | :----: | :---------: |
|        SRCC         |   0    |      0      |
|        KRCC         |   0    |      0      |
|        PLCC         |   0    |      0      |
|        RMSE         |   0    |      0      |

| Trained on YouTube-UGC | KoNViD | LIVE-VQC |
| :--------------------: | :----: | :------: |
|          SRCC          |   0    |    0     |
|          KRCC          |   0    |    0     |
|          PLCC          |   0    |    0     |
|          RMSE          |   0    |    0     |
