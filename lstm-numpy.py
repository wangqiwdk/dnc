# -*- coding: utf-8 -*-
# author: kmrocki

from __future__ import print_function
import numpy as np
import argparse, sys
import datetime, time
import random
from random import uniform


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', type=int, default = 16, help='batch size')
parser.add_argument('--hidden', type=int, default = 32, help='hiddens')
parser.add_argument('--seqlength', type=int, default = 25, help='seqlength')

T = 10 # max time

opt = parser.parse_args()
B = opt.batchsize # batch size
S = opt.seqlength # unrolling in time steps
HN = opt.hidden # size of hidden layer of neurons
print(f'B={B},S={S},HN={HN}')
learning_rate = 1e-1
clipgrads = False

# data I/O
data = open('./alice29.txt', 'r').read() # should be simple plain text file

chars = list(set(data))
data_size, M = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, M))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# controller parameters
Wxh = np.random.randn(4*HN, M)*0.01 # input to hidden
Whh = np.random.randn(4*HN, HN)*0.01 # hidden to hidden
Why = np.random.randn(M, HN)*0.01 # hidden to output
bh = np.zeros((4*HN, 1)) # hidden bias
by = np.zeros((M, 1)) # output bias

# init LSTM f gates biases higher
bh[2*HN:3*HN,:] = 1

def train(inputs, targets, cprev, hprev):
	"""
	inputs,targets are both list of integers.
	cprev is HxB array of initial memory cell state
	hprev is HxB array of initial hidden state
	returns the loss, gradients on model parameters, and last hidden state
	"""
	# xs输入向量(t,M,B) hs隐藏状态(t,HN,B) ys输出向量(t,M,B) ps预测字符向量(t,M,B)
	# gs门向量(t,4HN,B) cs cell状态(t,HN,B)
	# inputs, outputs, controller states
	xs, hs, ys, ps, gs, cs = {}, {}, {}, {}, {}, {}
	#init previous states
	# 初始化-1时刻状态
	hs[-1], cs[-1] = np.copy(hprev), np.copy(cprev)

	loss = 0
	# 前向计算，从序列第一个位置开始
	# forward pass
	for t in range(len(inputs)):
		# t时刻输入字符的one-hot编码
		xs[t] = np.zeros((M, B)) # encode in 1-of-k representation
		for b in range(0,B): xs[t][:,b][inputs[t][b]] = 1
		
		# 计算门向量 
		# gs[t]形状为(4HN,B)
		# 其中
		# (0:HN,:)为输入门
		# (HN:2HN,:)为输出门
		# (2HN:3HN,:)为遗忘门
		# (3HN:4HN,B)为新cell状态
		gs[t] = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh # gates, linear part

		# gates nonlinear part
		gs[t][0:3*HN,:] = sigmoid(gs[t][0:3*HN,:]) #i, o, f gates
		gs[t][3*HN:4*HN, :] = np.tanh(gs[t][3*HN:4*HN,:]) #c gate

		# cell[t] = input_gate * new_cell + forget_gate * cell[t-1]
		# hide[t] = out_gate * tanh(cell[t])
		cs[t] = gs[t][3*HN:4*HN,:] * gs[t][0:HN,:] + gs[t][2*HN:3*HN,:] * cs[t-1]
		# NOTE: 是否保留tanh到下一轮？
		cs[t] = np.tanh(cs[t]) # mem cell - nonlinearity
		hs[t] = gs[t][HN:2*HN,:] * cs[t] # new hidden state
		# 计算下一个字符的概率
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars

		###################
		# 归一化概率
		mx = np.max(ys[t], axis=0)
		ys[t] -= mx # normalize
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars

		# 累计batch中每个样本的损失
		for b in range(0,B):
			if ps[t][targets[t,b],b] > 0: loss += -np.log(ps[t][targets[t,b],b]) # softmax (cross-entropy loss)

	# 后向
	# backward pass:
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dcnext = np.zeros_like(cs[0])
	dhnext = np.zeros_like(hs[0])
	dg = np.zeros_like(gs[0])

	for t in reversed(range(len(inputs))):
		dy = np.copy(ps[t])
		for b in range(0,B): dy[targets[t][b], b] -= 1 # backprop into y
		dWhy += np.dot(dy, hs[t].T)

		dby += np.expand_dims(np.sum(dy,axis=1), axis=1)
		dh = np.dot(Why.T, dy) + dhnext # backprop into h

		dc = dh * gs[t][HN:2*HN,:] + dcnext # backprop into c
		dc = dc * (1 - cs[t] * cs[t]) # backprop though tanh

		dg[HN:2*HN,:] = dh * cs[t] # o gates
		dg[0:HN,:] = gs[t][3*HN:4*HN,:] * dc # i gates
		dg[2*HN:3*HN,:] = cs[t-1] * dc # f gates
		dg[3*HN:4*HN,:] = gs[t][0:HN,:] * dc # c gates
		dg[0:3*HN,:] = dg[0:3*HN,:] * gs[t][0:3*HN,:] * (1 - gs[t][0:3*HN,:]) # backprop through sigmoids
		dg[3*HN:4*HN,:] = dg[3*HN:4*HN,:] * (1 - gs[t][3*HN:4*HN,:] * gs[t][3*HN:4*HN,:]) # backprop through tanh
		dbh += np.expand_dims(np.sum(dg,axis=1), axis=1)
		dWxh += np.dot(dg, xs[t].T)
		dWhh += np.dot(dg, hs[t-1].T)
		dhnext = np.dot(Whh.T, dg)
		dcnext = dc * gs[t][2*HN:3*HN,:]

		if clipgrads:
			for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
				np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return loss, dWxh, dWhh, dWhy, dbh, dby, cs[len(inputs)-1], hs[len(inputs)-1]

n = 0 # 迭代计数
p = np.random.randint(len(data)-1-S,size=(B)).tolist()
inputs = np.zeros((S,B), dtype=int) # 输入向量(S,B)
targets = np.zeros((S,B), dtype=int) # 目标向量(S,B)
cprev = np.zeros((HN,B)) # cell状态向量(HN,B)
hprev = np.zeros((HN,B)) # 隐藏状态向量(HN,B)
# 用于adagrad算法，累计梯度
mWxh, mWhh, mWhy  = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/M)*S # loss at iteration 0 初始损失
start = time.time()

t = time.time()-start
last=start
while t < T:
	# prepare inputs (we're sweeping from left to right in steps S long)
	for b in range(0,B):
		# 如果当前序列的起点超过最大起点，或者当前为第一轮，
		# 则重置对应序列的cprev[:,b]、hprev[:,b]
		# 并重新选择当前序列的起点
		if p[b]+S+1 >= len(data) or n == 0:
			cprev[:,b] = np.zeros(HN) # reset LSTM memory
			hprev[:,b] = np.zeros(HN) # reset hidden memory
			p[b] = np.random.randint(len(data)-1-S)
		# 构建新一轮的输入与目标向量
		inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+S]]
		targets[:,b] = [char_to_ix[ch] for ch in data[p[b]+1:p[b]+S+1]]

	# 前向计算S个字符，返回损失与各参数的梯度
	# forward S characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, cprev, hprev = train(inputs, targets, cprev, hprev)
	smooth_loss = smooth_loss * 0.999 + np.mean(loss)/(np.log(2)*B) * 0.001
	if n % 10 == 0:
		tdelta = time.time()-last
		last = time.time()
		t = time.time()-start
		print('%.3f s, iter %d, %.4f BPC, %.2f char/s' % (t, n, smooth_loss / S, (B*S*10)/tdelta)) # print progress
	
	# 更新参数
	for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
									[dWxh, dWhh, dWhy, dbh, dby], 
									[mWxh, mWhh, mWhy, mbh, mby]):
		# perform parameter update with Adagrad
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	# 本轮数据的序列终点，作为下轮的序列起点
	for b in range(0,B): p[b] += S # move data pointer
	n += 1 # iteration counter
