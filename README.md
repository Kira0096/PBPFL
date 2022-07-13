
Official repository for WWW 2022 paper **An Accuracy-Lossless Perturbation Method for Defending Privacy Attacks in Federated Learning**. 
This project is developed based on Python 3.6. 

## Install prerequisites
```
pip install -r requirements.txt
```

## Download datasets
Download the  "" [[download link]](https://drive.google.com/open?id=1vEyOsYi06-u2PF7vr-Hj4vzAqb7jX5h6) and run 'unzip dataset.zip' at the root directory before training.

# Regression experiments
All code for this part are included in "UMBD" and "ABD" subfolder. Please change to that folder before running the code.



## Running demos


### PBPFL training
* Train models with PBPFL

```
bash pbpfl_train.sh
```

### AVGFL training
* Train models with traditional AVGFL

```
bash avgfl_train.sh
```


# Classification experiments
All code for this part are included in "LDC" subfolder. Please change to that folder before running the code.


## Running demos

### PBPFL training
* Train models with PBPFL

```
bash pbpfl_train.sh
```

### AVGFL-MSE training
* Train models with traditional AVGFL on MSE loss

```
bash avgfl_mse_train.sh
```

### AVGFL training
* Train models with traditional AVGFL on CE loss

```
bash avgfl_ce_train.sh
```

## Citation
Please cite our paper in your publications if it helps your research:

```
@inproceedings{Yang_WWW_2022,
  title={An Accuracy-Lossless Perturbation Method for Defending Privacy Attacks in Federated Learning},
  author={Xue Yang, Yan Feng, Weijun Fang, Jun Shao, Xiaohu Tang, Shu-Tao Xia, Rongxing Lu},
  booktitle={Proceedings of the ACM Web Conference 2022},
  year={2022}
}
```



