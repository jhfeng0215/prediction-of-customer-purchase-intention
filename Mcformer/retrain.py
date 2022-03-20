import math
import re
import time

from torch import nn, optim
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from data import *
from models.model.transformer import Transformer

from util.epoch_timer import epoch_time


def load_record(path):
    f = open(path, 'r')
    losses = f.read()
    losses = re.sub('\\]', '', losses)
    losses = re.sub('\\[', '', losses)
    losses = re.sub('\\,', '', losses)
    losses = losses.split(' ')
    losses = [float(i) for i in losses]
    return losses, len(losses)


def load_weight(model):
    model.load_state_dict(torch.load("./saved/model-0.8232114911079407.pt"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_losses, train_count = load_record('./result/train_loss.txt')
test_losses, _ = load_record('./result/test_loss.txt')
epoch -= train_count

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_cat_sos_idx=trg_cat_sos_idx,
                    trg_type_sos_idx=trg_type_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
load_weight(model=model)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (X_train_en, X_train_de,y_train) in enumerate(iterator):
        src = X_train_en
        trg = X_train_de

        optimizer.zero_grad()

        output = model(src, trg[:, : ,:])   #[32, 2]
        y_train=nn.functional.one_hot(y_train, 2)
        y_train =torch.tensor(y_train,dtype=torch.float32)
        # output = torch.tensor(output.numpy())
        loss = criterion(output, y_train)
        # print('-------loss----{}'.format(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():

        for i, (X_test_en, X_test_de,y_test)  in enumerate(iterator):
            src = X_test_en
            trg = X_test_de
            output = model(src, trg[:, : ,:])
            y_test_oh = nn.functional.one_hot(y_test, 2)
            y_test_oh = torch.tensor(y_test_oh, dtype=torch.float32)
            loss = criterion(output, y_test_oh)
            epoch_loss += loss.item()

            _,pre_lab=torch.max(output,1)
            test_accuracy=accuracy_score(y_test,pre_lab)
            test_f1=f1_score(y_test,pre_lab)
            print('test_accuracy--{}|f1--{}'.format(test_accuracy,test_f1))


    return epoch_loss / len(iterator)


def run(total_epoch, best_loss):
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss= evaluate(model, test_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()


        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1 + train_count} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')



if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
