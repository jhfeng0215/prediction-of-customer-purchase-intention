
# Mcformer---An end to end model to predict customer-purchase-intention

发明人： Jiahui Feng(冯家辉), Qihang Zhao(赵启航), Hefu Liu(刘和福)

* This algorithm was developed by members of the IOM Lab at the School of Management, USTC
* This is an end to end deep learning method Mcformer to utilize the customer clickstream data to predict the user purchase intention.
* We aim to utilize the customer clickstream data to predict the customer purchase intention, the scenes as follows:
* ![question](https://github.com/jhfeng0215/prediction-of-customer-purchase-intention/blob/main/Mcformer/pictures/ques.png)

## The framework of Mcformer
![Framework of Mcformer](https://github.com/jhfeng0215/prediction-of-customer-purchase-intention/blob/main/Mcformer/pictures/modelframe.png)
* Introduction of Mcformer
*  In order to deal with multi-dimension clickstream sequence data, we proposed an end-to-end deep learning model, named Multi-channel for purchase transformer (Mcformer), to predict the customers’ purchasing intention. Figure 1 shows the model architecture of Mcformer. This model composed by four parts: embedding layer, multi-transformer layer, cross fusion layer and output layer. Embedding layer is used to embed the sparse one-hot vectors of the behavior data to dense vectors. After that, multi-channels transformer identify intra-information of each sequence. Then the cross fusion layer is applied to identify the inter-information of different sequences. Finally, Mcformer output the result by the multilayer perceptron. 

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
* ![data](https://github.com/jhfeng0215/prediction-of-customer-purchase-intention/blob/main/Mcformer/pictures/datades.png)

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

## The results
* The result show that Mcformer get great performance in long sequence classification tasks.
* The beat parameters in the file named optimizpara, so you can use our best parameters directly.

## Parameter
* We have disclosed the model best parameters, due to the large file size, you can click the hyperlink to download:
* HyperLink:https://pan.baidu.com/s/10pkew7_tZgbdbjGm0XxDbA verification code：pxxk











