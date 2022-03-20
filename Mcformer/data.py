"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
import pandas as pd
from modelconf import *


def str2int(x):
    a = []
    for i in x:
        a.append(int(i))
    return a


df=pd.read_csv(r'/root/autodl-tmp/feizhu_data/split/train_80_100.csv',engine='python')
# df=pd.read_csv(r'/root/autodl-nas/taobao7sample/train_60_80.csv',engine='python')
#category_seq,type_seq
#[ 'useid', 'itemid', 'category', 'type', 'time', 'label']
df= df.sample(frac=1)
data_list=[]
for other in ['itemid', 'category', 'type']:

    try:
        other_seq = df[other].apply(lambda x: eval(x)).values
        type_seq = df['hour'].apply(lambda x: eval(x)).values
        time_interval = df['minute'].apply(lambda x: pd.Series(eval(x)).diff(1).shift(-1).fillna(0).astype(int).to_list()).values
    except:
        other_seq = df[other].apply(lambda x: eval(x)).values
        type_seq = df['hour'].apply(lambda x: eval(x)).values
        time_interval = df['minute'].apply(lambda x: pd.Series(eval(x)).diff(1).shift(-1).fillna(0).to_list()).values

    pad_unk=''
    if other=='itemid':
        pad_unk=item_pad_unk
    elif other=='category':
        pad_unk= cat_pad_unk
    elif other =='type':
        pad_unk = type_pad_unk
    label = df['label'].values


    loader = DataLoader(max_len)

    X_train_en,X_test_en,X_train_de,X_test_de,y_train,y_test,last_wordlist= loader.make_dataset(other_seq,type_seq,time_interval,label,pad_unk)
    data_list.append([X_train_en,X_test_en,X_train_de,X_test_de,y_train,y_test,last_wordlist])

train_iter, valid_iter = loader.make_iter(data_list,batch_size=batch_size,device=device)


#读取数据
if is_test:
    #修改路径


    df=pd.read_csv(r'/root/autodl-tmp/feizhu_data/split/test_80_100.csv',engine='python')
    data_list = []
    for other in ['itemid', 'category', 'type']:
        try:
            other_seq = df[other].apply(lambda x: eval(x)).values
            type_seq = df['hour'].apply(lambda x: eval(x)).values
            time_interval = df['minute'].apply(lambda x: pd.Series(eval(x)).diff(1).shift(-1).fillna(0).astype(int).to_list()).values
        except:
            other_seq = df[other].apply(lambda x: eval(x)).values
            type_seq = df['hour'].apply(lambda x: eval(x)).values
            time_interval = df['minute'].apply(lambda x: pd.Series(eval(x)).diff(1).shift(-1).fillna(0).astype(int).to_list()).values

        pad_unk = ''
        if other == 'itemid':
            pad_unk = item_pad_unk
        elif other == 'category':
            pad_unk = cat_pad_unk
        elif other == 'type':
            pad_unk = type_pad_unk

        loader = DataLoader(max_len)
        label = df['label'].apply(lambda x: int(x)).values.tolist()
        category_en,category_de,last_wordlist = loader.test(other_seq,type_seq,time_interval,label,pad_unk)
        data_list.append([category_en, category_de, last_wordlist,label])

    test_iter = loader.test_iterator(data_list, batch_size=batch_size, device=device)
else:
    pass






