"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
from models.model.fcnn import Fcnn
import numpy as np
from torch import nn, optim
from torch.optim import Adam
from torch.optim import Adamax
from data import *
from models.model.transformer import Transformer
from util.epoch_timer import epoch_time
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

model = Fcnn(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    cat_pad_idx=0,
    cat_time_sos_idx = cat_time_sos_idx,
    cat_type_sos_idx = cat_type_sos_idx,
    item_pad_idx = 0,
    item_time_sos_idx = item_time_sos_idx,
    item_type_sos_idx = item_type_sos_idx,
    type_pad_idx = 0,
    type_time_sos_idx = type_time_sos_idx,
    type_type_sos_idx = type_type_sos_idx,
    max_len=max_len,
    d_model = d_model,
    d_model_1=d_model_1,
    ffn_hidden=ffn_hidden,
    n_heads=n_heads,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device,
).to(device)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
# torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# torch.optim.SparseAdam(params,lr=0.001,betas=(0.9,0.999),eps=1e-08)   #处理稀疏张量

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
# criterion = nn.CrossEntropyLoss()
criterion =nn.NLLLoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()  #为什么还有一个model.train
    epoch_loss = 0
    train_accuracy=[]
    train_f1=[]
    train_recall=[]
    train_precision=[]
    train_roc= []
    for i, (X_item_train_en, X_item_train_de, X_cat_train_en, X_cat_train_de,X_type_train_en,X_type_train_de,y_train,X_train_wlen) in enumerate(iterator):
        #encoding:[[产品序列],[点击序列] decoding[[产品序列],[点击序列]]
        optimizer.zero_grad()
        output = model(X_item_train_en, X_item_train_de, X_cat_train_en, X_cat_train_de,X_type_train_en,
            X_type_train_de,X_train_wlen)   #[32, 2]
        # y_train_oh =nn.functional.one_hot(y_train, 2)
        # y_train_oh =torch.tensor(y_train_oh,dtype=torch.float32)
        loss = criterion(output,y_train)

        # print('-------loss----{}'.format(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        # print('y_train_oh{}'.format(y_train_oh))
        # print(output)
        _, pre_lab = torch.max(output, 1)
        # print(pre_lab)
        y_train=y_train.data.cpu().numpy()
        pre_lab=pre_lab.data.cpu().numpy()
        train_accuracy.append(accuracy_score(y_train, pre_lab))
        train_f1.append(f1_score(y_train, pre_lab))
        train_recall.append(recall_score(y_train, pre_lab))
        train_precision.append(precision_score(y_train, pre_lab))
        try:
            train_roc.append(roc_auc_score(y_train, pre_lab))
        except:
            train_roc.append(0.5)
    train_accuracy=np.array(train_accuracy).mean()
    train_f1 = np.array(train_f1).mean()
    train_recall = np.array(train_recall).mean()
    train_precision = np.array(train_precision).mean()
    train_roc = np.array(train_roc).mean()
        # print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item(),', acc_score:', train_accuracy,', precision_score :', test_precision,', recall_score :', train_recall,', f1_score :', train_f1)

    return epoch_loss / len(iterator),train_accuracy,train_f1,train_recall,train_precision,train_roc


def valid(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    loss_list=[]
    acc_list=[]
    f1_list=[]
    recall_list=[]
    precision_list=[]
    roc_list=[]
    with torch.no_grad():
        for i, (X_item_test_en, X_item_test_de, X_cat_test_en, X_cat_test_de,X_type_test_en,X_type_test_de,
                y_test,X_test_wlen)  in enumerate(iterator):

            output = model(X_item_test_en, X_item_test_de, X_cat_test_en, X_cat_test_de,X_type_test_en,X_type_test_de,
                X_test_wlen)
            # y_test_oh = nn.functional.one_hot(y_test, 2)
            # y_test_oh = torch.tensor(y_test_oh, dtype=torch.float32)
            loss = criterion(output,y_test)
            epoch_loss += loss.item()

            _,pre_lab=torch.max(output,1)
            y_test = y_test.data.cpu().numpy()
            pre_lab = pre_lab.data.cpu().numpy()
            test_accuracy=accuracy_score(y_test,pre_lab)
            test_f1=f1_score(y_test,pre_lab)
            test_recall=recall_score(y_test,pre_lab)
            test_precision=precision_score(y_test,pre_lab)
            try:
                test_roc=roc_auc_score(y_test,pre_lab)
            except:
                test_roc=0.5
            # print('test_acc_score:{} ,test_f1_score:{} ,recall_score:{} ,test_precision{}'.format(test_accuracy,test_f1,test_recall, test_precision))
            roc_list.append(test_roc)
            loss_list.append(loss.item())
            acc_list.append(test_accuracy)
            f1_list.append(test_f1)
            recall_list.append(test_recall)
            precision_list.append(test_precision)
        test_accuracy=np.array(acc_list).mean()
        test_f1 = np.array(f1_list).mean()
        test_recall = np.array(recall_list).mean()
        test_precision = np.array(precision_list).mean()
        test_loss = np.array(loss_list).mean()
        test_roc = np.array(roc_list).mean()
        # target_names = ['label 0', 'label 1']
        # print(classification_report(y_true, y_pred, target_names=target_names))
        # print('----------------------test collection---------------------------------')

        # print('test_loss:{} ,test_acc_score:{} ,test_f1_score:{} ,recall_score:{} ,test_precision{}'.format(test_loss,test_accuracy,test_f1,test_recall,test_precision))
    return epoch_loss / len(iterator),test_accuracy,test_f1,test_recall,test_precision,test_roc


def tes(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    loss_list=[]
    acc_list=[]
    f1_list=[]
    recall_list=[]
    precision_list=[]
    roc_auc_score_list=[]
    with torch.no_grad():
        
        for i, (X_item_test_en, X_item_test_de, X_cat_test_en, X_cat_test_de,X_type_test_en,X_type_test_de,y_test,last_wordlist)  in enumerate(iterator):
            output = model(X_item_test_en, X_item_test_de, X_cat_test_en, X_cat_test_de,X_type_test_en,X_type_test_de,last_wordlist)
            # y_test_oh = nn.functional.one_hot(y_test, 2)
            # y_test_oh = torch.tensor(y_test_oh, dtype=torch.float32)
            loss = criterion(output,y_test)
            epoch_loss += loss.item()

            _,pre_lab=torch.max(output,1)
            y_test = y_test.data.cpu().numpy()
            pre_lab = pre_lab.data.cpu().numpy()
            test_accuracy=accuracy_score(y_test,pre_lab)
            test_f1=f1_score(y_test,pre_lab)
            test_recall=recall_score(y_test,pre_lab)
            test_precision=precision_score(y_test,pre_lab)
            try:
                roc_auc_score_list.append(roc_auc_score(y_test,pre_lab))
            except:
                roc_auc_score_list.append(0.5)
            # print('test_acc_score:{} ,test_f1_score:{} ,recall_score:{} ,test_precision{}'.format(test_accuracy,test_f1,test_recall, test_precision))

            loss_list.append(loss.item())
            acc_list.append(test_accuracy)
            f1_list.append(test_f1)
            recall_list.append(test_recall)
            precision_list.append(test_precision)
        test_accuracy=np.array(acc_list).mean()
        test_f1 = np.array(f1_list).mean()
        test_recall = np.array(recall_list).mean()
        test_roc_auc_score=np.array(roc_auc_score_list).mean()
        test_precision = np.array(precision_list).mean()
        test_loss = np.array(loss_list).mean()
        
        print('Test Loss: %.3f,  Test Acc: %.3f, Test Pre: %.3f, Testing F1: %.3f, Testing Rec: %.3f, Testing Roc: %.3f,' % (
            test_loss,test_accuracy,test_precision,test_f1,test_recall,test_roc_auc_score,
        ))
        # target_names = ['label 0', 'label 1']
        # print(classification_report(y_true, y_pred, target_names=target_names))
        # print('----------------------test collection---------------------------------')

        # print('test_loss:{} ,test_acc_score:{} ,test_f1_score:{} ,recall_score:{} ,test_precision{}'.format(test_loss,test_accuracy,test_f1,test_recall,test_precision))
    return test_loss,test_accuracy,test_f1,test_recall,test_precision,test_roc_auc_score


def run(total_epoch, best_loss,is_test):
    train_losses, test_losses,train_f1s,test_f1s= [], [],[],[]
    for step in range(total_epoch):
        start_time = time.time()
        train_loss ,train_accuracy,train_f1,train_recall,train_precision,train_roc= train(model, train_iter, optimizer, criterion, clip)

        valid_loss,test_accuracy,test_f1,test_recall,test_precision ,test_roc= valid(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            # torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        f = open('result/train_f1.txt', 'w')
        f.write(str(train_f1s))
        f.close()

        f = open('result/test_f1.txt', 'w')
        f.write(str(test_f1s))
        f.close()
        # print('[Epoch: %3d/%3d] Training Loss: %.3f, Time:%.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f,Training Pre: %.3f, Testing Pre: %.3f,Training F1: %.3f, Testing F1: %.3f,Training Rec: %.3f, Testing Rec: %.3f'
        #       % (step + 1, total_epoch,epoch_mins+'m'+ epoch_secs+'s', train_loss, valid_loss, train_accuracy, test_accuracy,train_precision,test_precision,train_f1,test_f1,train_recall,test_recall))
        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f,Training Pre: %.3f, Testing Pre: %.3f,Training F1: %.3f, Testing F1: %.3f,Training Rec: %.3f, Testing Rec: %.3f'
            % (step + 1, total_epoch, train_loss, valid_loss, train_accuracy,
               test_accuracy, train_precision, test_precision, train_f1, test_f1, train_recall, test_recall))
        #
        # print(f'-------------------Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s------------------------------')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        if is_test:
            tes(model, test_iter, criterion)
        else:
            pass
        


import multiprocessing as mp
if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf,is_test = is_test)
    seed = 42
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
#     mp.set_start_method('spawn')

    # torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

