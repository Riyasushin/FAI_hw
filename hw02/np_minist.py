# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


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
    return (x > 0).astype(float)


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

    对角线 S(1 - S)
    非对角线 -S_i S_j

    x: N, C
    Return: 
        -  (N, C, C)
    '''
    # TODO
    softmax_output = f(x)  # 计算 Softmax 输出
    N, C = softmax_output.shape
    # 创建一个三维数组来存储每个样本的雅可比矩阵
    jacobian_matrix = np.zeros((N, C, C))
    # 创建一个三维单位矩阵，用于区分对角线和非对角线元素
    identity = np.eye(C)
    # 计算雅可比矩阵的每个元素
    for i in range(N):
        # 对角线元素：S_i (1 - S_i)
        jacobian_matrix[i] = np.diag(softmax_output[i] * (1 - softmax_output[i]))
        # 非对角线元素：-S_i S_j
        outer = -np.outer(softmax_output[i], softmax_output[i])
        jacobian_matrix[i] += outer
        # 利用单位矩阵将对角线元素恢复为正确值
        jacobian_matrix[i] += identity * softmax_output[i] * (1 - softmax_output[i])
    return jacobian_matrix


    

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    
    cross entropy: 
    '''
    epsilon = 1e-13
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出

    交叉熵损失对 Softmax 输入的导数
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
        Linear_out_1(l_1): N, hidden_size(H)
            ReLU:                               [drelu]    <- 
        ReLU_out(relu):   N, hidden_size(H)
            Linear hidden_size(H), output_size(C)     [dW2, db2] <- [d_linear_2]
        Linear_out_2(l_2): N, output_size(C)
            Softmax                             
        out: N, output_size
        '''
        self.W1 = init_weights((input_size, hidden_size))
        self.b1 = init_weights((hidden_size, ))
        self.W2 = init_weights((hidden_size, output_size))
        self.b2 = init_weights((output_size, ))
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
        cache['l1'] = np.matmul(x, self.W1) + self.b1 # N, H
        # ReLU
        # cache['relu'] = relu(cache['l1']) # N, H
        # cache['l2'] = np.matmul(cache['relu'], self.W2) + self.b2 # N, C
        # Sigmoid
        # cache['sigmoid'] = sigmoid(cache['l1'])
        # cache['l2'] = np.matmul(cache['sigmoid'], self.W2) + self.b2 # N, C
        # tanh
        cache['tanh'] = tanh(cache['l1'])
        cache['l2'] = np.matmul(cache['tanh'], self.W2) + self.b2 # N, C

        out = f(cache['l2'])
        return out, cache

    # TODO write backward()


    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''

        # 前向传播
        out, cache = self.forward(x_batch)
        
        # 计算损失和准确率
        loss = loss_fn(y_batch, out)
        chosen_indexs = np.argmax(out, axis=1) # N
        idx_first = [i for i in range(y_batch.shape[0])]
        acc_num = (y_batch[idx_first, chosen_indexs] == 1).astype(int)
        acc = acc_num.mean()
        # print(acc_num.shape) # (64, )
        
        # 反向传播

        d_linear_2 = loss_fn_prime(y_batch, out) # N, C

        # ReLU
        # dW2 = np.matmul(cache['relu'].T, d_linear_2) # X.T @ delta; H, C
        # Sigmoid
        # dW2 = np.matmul(cache['sigmoid'].T, d_linear_2)
        # Tanh
        dW2 = np.matmul(cache['tanh'].T, d_linear_2)

        db2 = np.sum(d_linear_2, axis=0)  # (C,)

        # ReLu
        # d_relu = np.matmul(d_linear_2, self.W2.T) # N, H
        # d_linear_1 = d_relu * relu_prime(cache['l1']) # N, H
        # Sigmoid
        # d_sigmoid = np.matmul(d_linear_2, self.W2.T) # N, H
        # d_linear_1 = d_sigmoid * sigmoid_prime(cache['l1']) # N, H
        # Tanh
        d_tanh = np.matmul(d_linear_2, self.W2.T) # N, H
        d_linear_1 = d_tanh * tanh_prime(cache['l1']) # N, H

        dW1 = np.matmul(x_batch.T, d_linear_1)  # (I, H)
        db1 = np.sum(d_linear_1, axis=0)  # (H,)

        batch_size = y_batch.shape[0]
        dW2 /= batch_size
        db2 /= batch_size
        dW1 /= batch_size
        db1 /= batch_size



        # 更新权重
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

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
    net = Network(input_size=784, hidden_size=128, output_size=10, lr=0.01)
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

            # 更新进度条 TODO
            # print(sum(losses) / cnt)
            # print(sum(accuracies) / cnt)
            p_bar.set_description(f'Epoch {epoch + 1},  Loss: {np.mean(losses):.4f},  Acc: {np.mean(accuracies):.4f}')

        val_loss, val_acc = net.evaluate(X_val, y_val, batch_size=64)
        print(f'Epoch {(epoch+1)}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} \
              lr: {net.lr} net-hiddensize:{net.W1.shape[1]} activation_func: tanh ')
    test_loss, test_acc = net.evaluate(X_test, y_test, batch_size=64)
    print(f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f} \
        lr: {net.lr} net-hiddensize:{net.W1.shape[1]} activation_func: tanh ')
    
        