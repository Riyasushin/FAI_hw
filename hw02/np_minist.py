# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm

# param
activation_func = 'tanh'


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(x, 0)
def relu_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    '''
    sigmoid函数
    '''
    return 1. / (1. + np.exp(-x))
def sigmoid_prime(x):
    '''
    sigmoid函数的导数
    '''
    return sigmoid(x) * (1. - sigmoid(x))

def tanh(x):
    '''
    tanh
    '''
    return np.tanh(x)
def tanh_prime(x):
    '''
    tanh函数的导数
    '''
    return 1 - np.tanh(x) ** 2


#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    x:  N, C
    '''
    max_x = np.max(x, axis=1, keepdims=True) # N,
    x_exp_moved = np.exp(x - max_x) # N, C
    return x_exp_moved / np.sum(x_exp_moved, axis=1, keepdims=True)

def f_prime(x):
    '''
    softmax函数的导数
    '''
    return f(x) * (1 - f(x))


    

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    
    cross entropy: 
    '''
    # TODO ???
    epsilon = 1e-13
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出

    交叉熵损失对 Softmax 输入的导数 !!!!!!
    '''
    return y_pred - y_true



# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        X:  N, input_size
            Linear input_size(I), hidden_size(H)      [dW1, db1] <- 
        z1: N, hidden_size(H)
            ReLU:                               [drelu]    <- 
        a1(relu):   N, hidden_size(H)
            Linear hidden_size(H), hidden_size(H)     [dW2, db2] <- [d_linear_2]
        z2: N, H
            ReLU
        a2(relu):   N, hidden_size(H)
            Linear hidden_size(H), output_size(C)
        z3: N, output_size(C)
        out: N, output_size
        '''
        # TODO ??? input_size + 1
        self.W1 = init_weights((input_size, hidden_size))
        self.b1 = init_weights((hidden_size, ))
        self.W2 = init_weights((hidden_size, hidden_size))
        self.b2 = init_weights((hidden_size, ))
        self.W3 = init_weights((hidden_size, output_size))
        self.b3 = init_weights((output_size, ))

        # self.W2 = init_weights((hidden_size, output_size))
        # self.b2 = init_weights((output_size, ))
        self.lr = lr

    def forward(self, x):
        '''
        前向传播
        Inputs:
            - x: N, I

        Return:
            - out: MLP的输出结构 N, output_size
            - cache: MLP的中间量, 用于SGD
        '''
        cache = {}
        cache['input'] = x
        cache['z1'] = np.matmul(x, self.W1) + self.b1
        if activation_func == 'relu':
            cache['a1'] = relu(cache['z1'])
        elif activation_func == 'sigmoid':
            cache['a1'] = sigmoid(cache['z1'])
        elif activation_func == 'tanh':
            cache['a1'] = tanh(cache['z1'])
            
        cache['z2'] = np.matmul(cache['a1'], self.W2) + self.b2 # N, H
        if activation_func == 'relu':
            cache['a2'] = relu(cache['z2'])
        elif activation_func == 'sigmoid':
            cache['a2'] = sigmoid(cache['z2'])
        elif activation_func == 'tanh':
            cache['a2'] = tanh(cache['z2'])
          
        cache['z3'] = np.matmul(cache['a2'], self.W3) + self.b3 # N, H
        cache['a3'] = f(cache['z3'])     
         
        # cache['z2'] = np.matmul(cache['a1'], self.W2) + self.b2 # N, H
        # cache['a2'] = f(cache['z2'])    

        out = cache['a3']
        return out, cache


    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        
        # 前向传播
        y_pred, cache = self.forward(x_batch)
        
        # 计算损失和准确率
        loss = loss_fn(y_batch, y_pred)

        chosen_indexs = np.argmax(y_pred, axis=1) # N
        true_indexs = np.argmax(y_batch, axis=1)
        acc = np.mean(chosen_indexs == true_indexs)
        # print(acc_num.shape) # (64, )
        
        # 反向传播
        d_a3 = loss_fn_prime(y_batch, y_pred) # N, C
        d_W3 = np.matmul(cache['a2'].T, d_a3)
        d_b3 = np.sum(d_a3, axis=0)

        # d_a2 = loss_fn_prime(y_batch, y_pred) # N, C
        # d_W2 = np.matmul(cache['a1'].T, d_a2)
        # d_b2 = np.sum(d_a2, axis=0)

        if activation_func == 'relu':
            d_a2 = np.matmul(d_a3, self.W3.T) * relu_prime(cache['z2'])
        elif activation_func == 'sigmoid':
            d_a2 = np.matmul(d_a3, self.W3.T) * sigmoid_prime(cache['z2'])
        elif activation_func == 'tanh':
            d_a2 = np.matmul(d_a3, self.W3.T) * tanh_prime(cache['z2'])
        d_W2 = np.matmul(cache['a1'].T, d_a2)
        d_b2 = np.sum(d_a2, axis=0)

        if activation_func == 'relu':
            d_a1 = np.matmul(d_a2, self.W2.T) * relu_prime(cache['z1'])
        elif activation_func == 'sigmoid':
            d_a1 = np.matmul(d_a2, self.W2.T) * sigmoid_prime(cache['z1'])
        elif activation_func == 'tanh':
            d_a1 = np.matmul(d_a2, self.W2.T) * tanh_prime(cache['z1'])
        d_W1 = np.matmul(cache['input'].T, d_a1)
        d_b1 = np.sum(d_a1, axis=0)

        # 更新权重
        batch_size = y_batch.shape[0]
        self.W3 -= self.lr * d_W3 / batch_size
        self.b3 -= self.lr * d_b3 / batch_size
        self.W2 -= self.lr * d_W2 / batch_size
        self.b2 -= self.lr * d_b2 / batch_size
        self.W1 -= self.lr * d_W1 / batch_size
        self.b1 -= self.lr * d_b1 / batch_size

        return loss.item(), acc

    def evaluate(self, X_val, y_val, batch_size):
        losses = []
        accuracies = []
        cnt = 0
        for i in range(0, len(X_val), batch_size):
            cnt += 1
            x_batch = X_val[i : i + batch_size]
            y_batch = y_val[i : i + batch_size]
        
            # 前向传播
            out, _ = self.forward(x_batch)
        
            # 计算损失和准确率
            loss = loss_fn(y_batch, out)
            chosen_indexs = np.argmax(out, axis=1) # N
            idx_first = [i for i in range(y_batch.shape[0])]
            acc_num = (y_batch[idx_first, chosen_indexs] == 1).astype(int)
            acc = acc_num.mean()
        
            losses.append(loss.item())
            accuracies.append(acc)
    
        return sum(losses) / cnt, sum(accuracies) / cnt


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=20, output_size=10, lr=0.01)
    for epoch in range(10):
        losses = []
        accuracies = []
        cnt = 0
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            cnt += 1

            st_id = i
            ed_id = i + 64

            x_batch = X_train[st_id : ed_id]
            y_batch = y_train[st_id : ed_id]

            loss, acc = net.step(x_batch, y_batch)
            losses.append(loss)
            accuracies.append(acc)

            # print(f'loss:{loss}, batch_acc:{acc}')

            # print(sum(losses) / cnt)
            # print(sum(accuracies) / cnt)
            p_bar.set_description(f'Epoch {epoch + 1},  Loss: {np.mean(losses):.4f},  Acc: {np.mean(accuracies):.4f}')

        val_loss, val_acc = net.evaluate(X_val, y_val, batch_size=64)
        print(f'Epoch {(epoch+1)}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} \
              lr: {net.lr} net-hiddensize:{net.W1.shape[1]} activation_func: {activation_func} ')
    test_loss, test_acc = net.evaluate(X_test, y_test, batch_size=64)
    print(f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f} \
        lr: {net.lr} net-hiddensize:{net.W1.shape[1]} activation_func: {activation_func} ')
    
        