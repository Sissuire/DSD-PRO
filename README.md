# DSD-PRO

PyTorch implementation of **UGC-VQA based on decomposition (dual-stream decompostion, DSD) and recomposition (progressively residual aggregation, PRO)**

A new version v2.0 has been updated. We enable the normalization.

---------------------

The work had been already shaped in 2021, but suffered a long-time peer-review procedure with `submit-major-reject & resubmit-major-minor`. And during these hard days, I have seen some work with almost the same ideas has been published. Please don't be surprised if you do so. This is good since I can see we're in the same direction on this journey.

## Overview

[None]

## Usage

The method is simple enough. 

- Download the pretrained submodels first (see in the folder `pretrained_model`);
- Run `demo_extract_first.py` to extract features from dual streams. 
- Run `demo_run_main.py` or `demo_run_interdataset.py` to get the intra/inter-dataset performance in KoNViD-1K, LIVE-VQC, and YouTube-UGC. 

We have provided the extracted features in KoNViD-1K, LIVE-VQC, and YouTube-UGC in the folder `./data/`. Any other dataset would be ok with the same procedure.

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
| SRCC | 0.8606 |  0.8749  |   0.8406    |
| KRCC | 0.6772 |  0.6942  |   0.6517    |
| PLCC | 0.8605 |  0.8620  |   0.8363    |
| RMSE | 0.0845 |  0.0880  |   0.0900    |


### Inter-dataset
The model uses 80%/20 for training/validation in one database, and is directly tested on the others. As we DO NOT utilize nonlinear regression, PLCC and RMSE in the following table seem less appropriate.

| Trained on KoNViD | LIVE-VQC | YouTube-UGC |
| :---------------: | :------: | :---------: |
|       SRCC        |  0.7927  |   0.5284    |
|       KRCC        |  0.5940  |   0.3630    |
|       PLCC        |  0.7992  |   0.5339    |
|       RMSE        |  0.1029  |   0.1533    |

| Trained on LIVE-VQC | KoNViD | YouTube-UGC |
| :-----------------: | :----: | :---------: |
|        SRCC         | 0.7544 |   0.4632    |
|        KRCC         | 0.5647 |   0.3161    |
|        PLCC         | 0.7469 |   0.4779    |
|        RMSE         | 0.1240 |   0.1864    |

| Trained on YouTube-UGC | KoNViD | LIVE-VQC |
| :--------------------: | :----: | :------: |
|          SRCC          | 0.7532 |  0.6525  |
|          KRCC          | 0.5580 |  0.4649  |
|          PLCC          | 0.7485 |  0.6697  |
|          RMSE          | 0.1113 |  0.1479  |

### Citation
If you are interested in the work, or find the code helpful, please cite our work

```
@ARTICLE{dsdpro,  
  author={Liu, Yongxu and Wu, Jinjian and Li, Leida and Dong, Weisheng and Shi, Guangming},  
   journal={IEEE Transactions on Circuits and Systems for Video Technology},   
   title={Quality Assessment of UGC Videos Based on Decomposition and Recomposition},   
   year={2023},
   volume={33},
   number={3},
   pages={1043-1054},
   doi={10.1109/TCSVT.2022.3209007}}
}
```

### Contact

Feel free to contact me via `yongxu.liu@stu.xidian.edu.cn` or `sissuire@163.com` if any question or bug.
