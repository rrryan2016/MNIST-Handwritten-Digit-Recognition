# MNSIT Handwritten Digit Recognition

The code is based on the assignmnt 2 in CS231n. 

The develop platform is Windows 10 + Python 3.7 + Pychram. 

## General Introduction

### Three-Layer Convolution Neural Network 

**INPUT -> CONV -> ReLU -> POOL -> FC -> ReLU -> FC -> OUT**


### Dataset Information - MNIST

A gray picture dataset, the structure of per graph is [28x28x1]. The height and width is 28, while only 1 color channel.

In default, 60,000 pictures for training, while 10,000 for test. 

![Examples of MNIST](https://i.loli.net/2019/01/17/5c408a3a0d48c.jpeg)

More information please click: 
http://yann.lecun.com/exdb/mnist/


### File Declaration 

|File Name|Introduction|
|:-:|:-:|
|main.py|Main python file to run|
|data_load.py|To parse and load data in MNIST_data|
|MNIST_data| Dataset, including 4 files downloaded from official webpage of MNIST|
|/support/classifiers/cnn.py|the structure of three-layer CNN|

## Tech Details

### Data Pre Processing 
Due to standard dataset, the size of picture is fixed. So, I don't need to worry much. Call get_MNIST_data() in data_load.py in the beigining of main.py. In order to formulate the data, I still minus mean for each image file
    
`mean_image = np.mean(X_train,axis=0)`

`X_train -= mean_image`

`X_val -= mean_image`

`X_test -= mean_image`

### Hyperparameter Setting 

|Hyerparameter|Value|
|:-:|:-:|
|stride (CONV)|1|
|Pad(CONV)|1|
|Size of filter(CONV)|3|
|Number of filters(CONV)|32|
|Size of filter(POOL)|2|
|Stride (POOL)|2|
|Bacth size|50|
|Learning rate|1e-4|

> All functions below can run in main.py. They are all separated in parts, please properly comment or uncomment. WARINNG: It will take around 2 hours to do a cross validation of a parameter. 

### Overfit Small Dataset

Before apply the CNN designed directly to the big dataset, I check whether it can overfit a small dataset at first, so as to make sure that it have the ability to learn. 

![Result of Small Dataset Overfit](https://i.loli.net/2019/01/17/5c408a6103e11.png)

The detailed data of a random run is here below:

|Epoch|Train Accuracy|Validation Accuracy|
|:-:|:-:|:-:|
|1|41%|25.45%|
|2|51%|36.6%|
|...|...|...|
|6|95%|66.4%|
|7|99%|67.5%|
|8|97%|68.9%|
|9|100%|68.18%|
|10|100%|66.7%|
|...|100%|...|


### Train and Test the Dataset


### Cross Validation

![Cross Validation](https://i.loli.net/2019/01/17/5c408a91083ae.jpg)

Use *Accuracy* as the evaluation method. 

Cause the size of train is 50,000, so I do 5-fold cross validation here. To find a better setting of `filter_size(k)`, `batch_size` and `learning_rate`.  

`k_choices = [1,3,5,7]`

`batch_size_choices = [30,50,100,200]`

`learning_rate_choices = [1e-1,1e-2,1e-4,1e-8,1e-16]`

The results of cross validation are as below, and the average value is calculated manually. 

1. filter_size 

|Filter Size|Validation Accuracy(in average of 5 folds)|
|:-:|:-:|
|1|94.518%|
|3|97.086%|
|5|97.438%|
|7|97.584%|


2. bacth_size 

|Batch Size|Validation Accuracy(in average of 5 folds)|
|:-:|:-:|
|30|97.106%|
|50|97.056%|
|100|96.662%|
|200|96.374%|

3. learning_rate 

The result is wrong, so this part need further analysis. I guess some problems in calculation, cause the data is so tiny.

中文叙述地址：https://blog.csdn.net/AAAAshin/article/details/86532986
