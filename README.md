

# Mcfromer---The model to predict customer-purchase-intention
This is an end to end deep learning method Mcformer to utilize the customer clickstream data to predict the user purchase intention.

## The framework of Mcformer


## Requirements 
* sklearn
* pandas
* pytorch
* cuda

## Run the project

''' bash
python train.py
'''

## Data
Our data is the real world data from  https://tianchi.aliyun.com/dataset/dataDetail?dataId=649, this dataset need to preprocessing, which need long time.
If you need the data to verify our model, you could contact with us jiahuifeng@ustc.mail.edu.cn

## modify files to ensure the code work
And if you want to use your data, you have to provide
* the number of items: ni
* the number of category:nc
* the number of types: nt
* the number of hour/minutes: nh

you need to modify the files as follows: 
### modelconf.py 
* cat_pad_unk = [[0,nc+1,nc+2], [0,nh+1,nh+2]]
* item_pad_unk =[[0,ni+1,ni+2], [0,nh+1,nh+2]]
* type_pad_unk =[[0,nt+1,nt+2], [0,nh+1,nh+2]]

### data.py 
the file location

### train.py
if you want to visualize the training process ,you should change parameters.











