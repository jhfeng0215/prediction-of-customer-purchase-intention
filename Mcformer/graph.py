"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        idx=[i for i  in range(0,50)]
        train_loss = read('./result/train_loss.txt')
        test_loss = read('./result/test_loss.txt')
        train_f1 = read('./result/train_f1.txt')
        test_f1 = read('./result/test_loss.txt')
        # print(train)
        plt.figure(figsize=(20,15))
        plt.plot(train_loss, 'r', label='train')
        plt.plot(test_loss, 'b', label='validation')
        plt.plot(train_f1, 'r', label='train')
        plt.plot(test_f1, 'b', label='validation')
        plt.legend(loc='lower left')
        for x, y in zip(idx, train_loss):
            plt.text(x, y, '%4.3f' % y)
        for x, y in zip(idx, test_loss):
            plt.text(x, y, '%4.3f' % y)
        for x, y in zip(idx, train_f1):
            plt.text(x, y, '%4.3f' % y)
        for x, y in zip(idx, test_f1):
            plt.text(x, y, '%4.3f' % y)



    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
