"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn.functional import pad
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as Data

class DataLoader:

    def __init__(self,seq_len):
        self.cut_len=seq_len   #128

        print('dataset initializing start')

    def make_dataset(self,category_seq,type_seq,time_interval,label,pad_unk):
        #[[]]
        #先解决pad 和 unk
        # r_pad,unk
        en_input=[]
        de_input=[]
        pad,r_pad,unk=pad_unk[0]
        category_en,category_de, last_wordlist= self.cut_pad(category_seq,pad,r_pad,unk)
        pad,r_pad,unk = pad_unk[1]
        type_en,type_de,_ = self.cut_pad(type_seq,pad,r_pad,unk)
        time_int_en,time_int_de,_=self.cut_pad(time_interval,0,0,0)
        for i in range(0,len(category_en)):
            en_input.append([category_en[i],type_en[i],time_int_en[i]])
            de_input.append([category_de[i],type_de[i],time_int_de[i]])
        #encode decode 维度一样吗
        label = label.tolist()

        #划分数据集
        # X_train_en,X_test_en,y_train,y_test = train_test_split(en_input,label,random_state=42)
        # X_train_de,X_test_de = train_test_split(de_input,random_state=42)
        X_train_en= en_input[0:int(len(last_wordlist)*0.9)]
        X_test_en = en_input[int(len(last_wordlist)*0.9):]
        y_train = label[0:int(len(last_wordlist)*0.9)]
        y_test = label[int(len(last_wordlist)*0.9):]
        X_train_de = de_input[0:int(len(last_wordlist)*0.9)]
        X_test_de = de_input[int(len(last_wordlist)*0.9):]


        return X_train_en,X_test_en,X_train_de,X_test_de,y_train,y_test,last_wordlist


    def cut_pad(self,x_list,pad,r_pad,unk):
        x_en_mtx = []
        x_de_mtx = []
        last_wordlist = []
        for x in x_list:
            x = list(x)
            # 点击数据是1-4
            # item数据是1-item.unique
            # pad =0
            # max_item=4162024  min_type=1
            # max_category = 9439  min_type=1
            # max_type=4 min_type=1

            if len(x) >= self.cut_len-1:  #-1是因为127  后面右移成为128
                x = x[-self.cut_len+1:-1]    # -1是因为最后一个是unk
                x_en = np.pad(x, ((0, 1)), 'constant', constant_values=(unk)) # type_unk:5

                x_de = np.pad(x_en, ((1, 0)), 'constant', constant_values=(r_pad)).tolist()   # type右移 type_pad=最后的编码
                x_en = np.pad(x_en, ((0, 1)), 'constant', constant_values=(pad)).tolist()
                last_wordlist.append(x_de.index(unk))
                x_en_mtx .append(x_en)   #
                x_de_mtx .append(x_de)
            else:
                #先unk
                x = x[:-1]
                x = np.pad(x, (0, 1), 'constant', constant_values=(unk))  # type_unk:5
                #再右移
                x_de = np.pad(x, (1, 0), 'constant', constant_values=(r_pad))
                #补齐
                # x_en = np.pad(x, (( self.cut_len - len(x)),0), 'constant', constant_values=(pad)).tolist()
                # x_de = np.pad(x_de, ((self.cut_len - len(x) - 1),0), 'constant', constant_values=(pad)).tolist()
                x_en = np.pad(x, (0, self.cut_len - len(x)), 'constant', constant_values=(pad)).tolist()
                x_de = np.pad(x_de, (0,self.cut_len - len(x) - 1), 'constant', constant_values=(pad)).tolist()
                last_wordlist.append(x_de.index(unk))
                x_en_mtx .append(x_en)
                x_de_mtx .append(x_de)
        # x_en_mtx=torch.tensor(x_en_mtx)
        # x_de_mtx=torch.tensor(x_de_mtx)
        # last_wordlist=torch.tensor(last_wordlist)
        return x_en_mtx, x_de_mtx,last_wordlist


    def make_iter(self,datalist,batch_size, device):
        #['itemid', 'category', 'type']
        #X_train_en,X_test_en,X_train_de,X_test_de,y_train,y_test,last_wordlist

        X_item_train_en,X_item_validat_en,X_item_train_de,X_item_validat_de,=torch.tensor(datalist[0][0],device=device),torch.tensor(datalist[0][1],device=device),torch.tensor(datalist[0][2],device=device),torch.tensor(datalist[0][3],device=device)
        X_cat_train_en,X_cat_validat_en,X_cat_train_de,X_cat_validat_de=torch.tensor(datalist[1][0],device=device),torch.tensor(datalist[1][1],device=device),torch.tensor(datalist[1][2],device=device),torch.tensor(datalist[1][3],device=device)
        X_type_train_en,X_type_validat_en,X_type_train_de,X_type_test_de,=torch.tensor(datalist[2][0],device=device),torch.tensor(datalist[2][1],device=device),torch.tensor(datalist[2][2],device=device),torch.tensor(datalist[2][3],device=device)
        y_item_train, y_item_validat,last_wordlist = torch.tensor(datalist[0][4],device=device),torch.tensor(datalist[0][5],device=device),torch.tensor(datalist[0][6],device=device)
        X_train_wlen=torch.tensor(last_wordlist[0:int(len(last_wordlist)*0.9)],device=device)
        X_validat_wlen=torch.tensor(last_wordlist[int(len(last_wordlist)*0.9):],device=device)



        train_data_nots = Data.TensorDataset(X_item_train_en, X_item_train_de, X_cat_train_en, X_cat_train_de,X_type_train_en,X_type_train_de,y_item_train,X_train_wlen)
        train_iterator = Data.DataLoader(
            dataset=train_data_nots,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last = True,
        )


        validat_data_nots = Data.TensorDataset(X_item_validat_en, X_item_validat_de, X_cat_validat_en, X_cat_validat_de,X_type_validat_en,X_type_test_de,y_item_validat,X_validat_wlen)
        validat_iterator = Data.DataLoader(
            dataset=validat_data_nots,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last = True,
        )



        #GPU转换

        return train_iterator, validat_iterator


    def test(self,category_seq,type_seq,time_interval,label,pad_unk):

        en_input=[]
        de_input=[]
        pad,r_pad,unk=pad_unk[0]
        category_en,category_de, last_wordlist= self.cut_pad(category_seq,pad,r_pad,unk)
        pad,r_pad,unk=pad_unk[1]
        type_en,type_de,_ = self.cut_pad(type_seq,pad,r_pad,unk)
        time_int_en,time_int_de,_=self.cut_pad(time_interval,0,0,0)
        for i in range(0,len(category_en)):
            en_input.append([category_en[i],type_en[i],time_int_en[i]])
            de_input.append([category_de[i],type_de[i],time_int_de[i]])
        #encode decode 维度一样吗

        return en_input,de_input,last_wordlist

    def test_iterator(self,datalist, batch_size, device):
        # [category_en, category_de, last_wordlist,label, last_wordlist]
        label = datalist[0][3]
        label = torch.tensor(label, device=device)
        last_wordlist = torch.tensor(datalist[0][2], device=device)
        X_item_test_en,X_cat_test_en,X_type_test_en=torch.tensor(datalist[0][0],device=device),torch.tensor(datalist[1][0],device=device),torch.tensor(datalist[2][0],device=device)
        X_item_test_de,X_cat_test_de,X_type_test_de=torch.tensor(datalist[0][1],device=device),torch.tensor(datalist[1][1],device=device),torch.tensor(datalist[2][1],device=device)



        test_data_nots = Data.TensorDataset(X_item_test_en, X_item_test_de, X_cat_test_en, X_cat_test_de,X_type_test_en,X_type_test_de,label,last_wordlist)
        test_iterator = Data.DataLoader(
            dataset=test_data_nots,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last = True,
        )
        return test_iterator
