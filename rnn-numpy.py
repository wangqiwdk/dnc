# -*- coding: utf-8 -*-
# author: kmrocki
# based on the original code by A.Karpathy (char-rnn) https://gist.github.com/karpathy/d4dee566867f8291f086

import numpy as np
import argparse, sys
import datetime, time
import pickle
from random import uniform


### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', type=int, default = 16, help='batch size')
parser.add_argument('--hidden', type=int, default = 32, help='hiddens')
parser.add_argument('--seqlength', type=int, default = 25, help='seqlength')

opt = parser.parse_args()
B = opt.batchsize # batch size
S = opt.seqlength # unrolling in time steps
HN = opt.hidden # size of hidden layer of neurons
learning_rate = 1e-1
clipgrads = False

# data I/O
data = open('./样本数据.txt', 'r', encoding='utf-8').read()

chars = list(set(data)) # 获取可用字符集
with open('words.txt','w', encoding='utf-8') as f:
  f.write('\n'.join(chars))

data_size, M = len(data), len(chars) # 数据长度与字符集数量
print('data has %d characters, %d unique.' % (data_size, M))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# controller parameters 初始化参数
Wxh = np.random.randn(HN, M)*0.01 # input to hidden
Whh = np.random.randn(HN, HN)*0.01 # hidden to hidden
Why = np.random.randn(M, HN)*0.01 # hidden to output
bh = np.zeros((HN, 1)) # hidden bias
by = np.zeros((M, 1)) # output bias

def train(inputs, targets, hprev):
  """
  inputs (S,B) 输入序列
  targets (S,B) 目标序列
  hprev (HN,B) 隐藏状态

  inputs,targets are both list of integers.
  hprev is HxB array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  # xs输入向量(t,M,B) hs隐藏状态(t,HN,B) ys输出向量(t,M,B) ps预测字符向量(t,M,B)
  # inputs, outputs, controller states
  xs, hs, ys, ps = {}, {}, {}, {}
  # 初始隐藏状态标记为-1
  #init previous states
  hs[-1] = np.copy(hprev)

  loss = 0
  # 前向 遍历序列中每一个字符，即从时间0计算到时间S
  # forward pass
  for t in range(len(inputs)):
    # xs[t]保存每个序列中第t个字符的one-hot向量
    xs[t] = np.zeros((M, B)) # encode in 1-of-k representation
    for b in range(0,B): xs[t][:,b][inputs[t][b]] = 1

    # 由xs[t]和hs[t-1]计算t时刻隐藏状态
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    # 由hs[t]计算ys[t]
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars

    ###################
    # 将ys[t]归一化
    mx = np.max(ys[t], axis=0)
    ys[t] -= mx # normalize
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars

    # 累计所有序列的交叉熵损失
    for b in range(0,B):
        if ps[t][targets[t,b],b] > 0: loss += -np.log(ps[t][targets[t,b],b]) # softmax (cross-entropy loss)

  # 后向传播
  # backward pass:
  # 初始化用于记录梯度
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  # 倒序遍历序列，累计梯度
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t]) # (M,B)
    for b in range(0,B): dy[targets[t][b], b] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T) # (M,B)*(B,HN)=(M,HN)
    dby += np.expand_dims(np.sum(dy,axis=1), axis=1)
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dh = dh * (1 - hs[t] * hs[t]) # backprop though tanh
    dbh += np.expand_dims(np.sum(dh,axis=1), axis=1)
    dWxh += np.dot(dh, xs[t].T)
    dWhh += np.dot(dh, hs[t-1].T)
    dhnext = np.dot(Whh.T, dh)

    if clipgrads:
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((M, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(M), p=p.ravel())
    x = np.zeros((M, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
  
n = 0 # 迭代计数器
# p为用于一个batch的随机样本数据
# len(data)-1-S 代表输入序列的最大索引，即总长度-1-序列长度S
p = np.random.randint(len(data)-1-S,size=(B)).tolist() 
inputs = np.zeros((S,B), dtype=int)
targets = np.zeros((S,B), dtype=int)
hprev = np.zeros((HN,B))
# 用于adagrad算法，累计梯度
mWxh, mWhh, mWhy  = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/M)*S # loss at iteration 0
start = time.time()

t = time.time()-start
last=start
T = 1000 # max time
while t < T:
  # 构建新一轮的输入与目标向量
  # prepare inputs (we're sweeping from left to right in steps S long)
  for b in range(0,B):
      # 如果当前序列的起点超过最大起点，或者当前为第一轮，则初始化前一轮隐藏层hprev和当前序列的起点
      if p[b]+S+1 >= len(data) or n == 0:
        hprev[:,b] = np.zeros(HN) # reset hidden memory
        p[b] = np.random.randint(len(data)-1-S)
      # 从data的第p[b]个字符开始获取长度为S的序列，作为一个样本，shape=(S,1)
      inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+S]]
      # 从data的第p[b]+1个字符开始长度为S的序列，作为目标向量，shape=(S,1)
      targets[:,b] = [char_to_ix[ch] for ch in data[p[b]+1:p[b]+S+1]]

  # 本批次数据输入train函数，返回损失、各参数梯度、隐藏状态矩阵
  # forward S characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = train(inputs, targets, hprev)

  # 损失更新
  smooth_loss = smooth_loss * 0.999 + np.mean(loss)/(np.log(2)*B) * 0.001
  # 每10轮，输出一次进度
  if n % 10 == 0:
      tdelta = time.time()-last
      last = time.time()
      t = time.time()-start
      print('%.3f s, iter %d, %.4f BPC, %.2f char/s' % (t, n, smooth_loss / S, (B*S*10)/tdelta)) # print progress
  
  if n % 100 == 0:
    sent=sample(np.zeros((HN,1)), chars.index('我'), 100)
    sent=[chars[i] for i in sent]
    print('我'+''.join(sent))

  # 使用adagrad算法更新参数
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    # perform parameter update with Adagrad
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  # 本轮数据的序列终点，作为下轮的序列起点
  for b in range(0,B): p[b] += S # move data pointer
  n += 1 # iteration counter