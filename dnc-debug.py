# -*- coding: utf-8 -*-
# author: kmrocki

import argparse
import datetime
import sys
import time
from random import uniform
from typing import Type, List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from scipy.special import expit as sigmoid  # def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
from scipy.special import logsumexp

array: Type[ndarray] = np.ndarray
floatarray: Type[list] = list[Type[float]]
listOfArrays: Type[list] = list[Type[ndarray]]

def dfun_key_simil(C, dsim): return np.dot(C.T, dsim)

def mag(V: array): return np.sqrt(np.sum(V * V))

def normalize(V: array): return V / mag(V)


### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fname', type=str, default=sys.argv[0] + '.log', help='log filename')
parser.add_argument('--batchsize', type=int, default=16, help='batch size')
parser.add_argument('--hidden', type=int, default=64, help='hiddens')
parser.add_argument('--seqlength', type=int, default=25, help='seqlength')
parser.add_argument('--timelimit', type=int, default=600, help='time limit (s)')
parser.add_argument('--gradcheck', action='store_const', const=True, default=False, help='run gradcheck?')
parser.add_argument('--fp64', action='store_const', const=True, default=False, help='double precision?')
parser.add_argument('--sample_length', type=int, default=500, help='sample length')
parser.add_argument('--report_interval', type=int, default=200, help='report interval (sample, grads)')

opt = parser.parse_args()
print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt))
logname = opt.fname
gradchecklogname = 'gradcheck.log'
samplelogname = 'sample.log'
B:int = opt.batchsize
S:int = opt.seqlength
T = opt.timelimit
GC = opt.gradcheck
plotting = False

val = np.float32 if opt.fp64 else np.float64

clipGradients = False
learning_rate = 1e-1  # 5*1e-2
# TODO check the size constraints with regard to HN
MW:int = 8  # paper - W
MN:int = 8  # paper - N
MR:int = 1  # paper - R

def key() -> array: return np.zeros((MW, B), dtype=val)
def gate() -> array: return np.zeros((MN, B), dtype=val)

# hyperparameters
N:int = opt.hidden  # size of hidden layer of neurons
S:int = opt.seqlength  # number of steps to unroll the RNN for
B:int = opt.batchsize
inputs:array = np.zeros((S, B), dtype=int)
I:int = len(inputs)

# gradient checking
def gradCheck(cprev, hprev, mprev, rprev):
    global Wxh, Whh, Why, Whr, Whv, Whw, Whe, Wrh, bh, by
    num_checks, delta = 10, 1e-5
    _, dbh, dby, _, _ = learn(cprev, hprev, mprev, rprev)
    #print('GRAD CHECK\n')
    with open(gradchecklogname, "w") as myfile:
        myfile.write("-----\n")

    for param, dparam, name in zip([Wxh, Whh, Why, Whr, Whv, Whw, Whe, Wrh, Wry, bh, by],
                                   [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby],
                                   ['Wxh', 'Whh', 'Why', 'Whr', 'Whv', 'Whw', 'Whe', 'Wrh', 'Wry', 'bh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (repr(s0), repr(s1))
        min_error, mean_error, max_error = 1, 0, 0
        min_numerical, max_numerical = 1e10, -1e10
        min_analytic, max_analytic = 1e10, -1e10
        valid_checks = 0
        for i in range(num_checks):
            ri:int = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _, _, _, _, _, _, _, _ = learn(cprev, hprev, mprev, rprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _, _, _, _, _, _, _, _ = learn(cprev, hprev, mprev, rprev)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)

            vdiff = abs(grad_analytic - grad_numerical)
            vsum = abs(grad_numerical + grad_analytic)
            min_numerical = min(grad_numerical, min_numerical)
            max_numerical = max(grad_numerical, max_numerical)
            min_analytic = min(grad_analytic, min_analytic)
            max_analytic = max(grad_analytic, max_analytic)

            if vsum > 0:
                rel_error = vdiff / vsum
                min_error = min(min_error, rel_error)
                max_error = max(max_error, rel_error)
                mean_error = mean_error + rel_error
                valid_checks += 1

        mean_error /= num_checks
        print('%s:\t\tn = [%e, %e]\tmin %e, max %e\t\n\t\ta = [%e, %e]\tmean %e # %d/%d' % (
        name, min_numerical, max_numerical, min_error, max_error, min_analytic, max_analytic, mean_error, num_checks,valid_checks))
        # rel_error should be on order of 1e-7 or less
        entry = '%s:\t\tn = [%e, %e]\tmin %e, max %e\t\n\t\ta = [%e, %e]\tmean %e # %d/%d\n' % (
        name, min_numerical, max_numerical, min_error, max_error, min_analytic, max_analytic, mean_error, num_checks,valid_checks)
        with open(gradchecklogname, "a") as myfile:
            myfile.write(entry)


start = time.time()
with open(logname, "a") as myfile:
    entry = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt
    myfile.write("# " + str(entry))
    myfile.write("\n#  ITER\t\tTIME\t\tTRAIN LOSS\n")

# data = open('./ptb/ptb.train.txt', 'r').read() # should be simple plain text file
f = open('./alice29.txt', 'r')
data = f.read()
f.close()

chars = list(set(data))
data_size : int = len(data)
vocab_size : int = len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix: dict = {ch: i for i, ch in enumerate(chars)}
ix_to_char: dict = {i: ch for i, ch in enumerate(chars)}

# model parameters
rngAmp:float = 0.01

Wxh : array = np.random.randn(4 * N, vocab_size).astype(val) * rngAmp  # input to hidden
Whh : array = np.random.randn(4 * N, N).astype(val) * rngAmp  # hidden to hidden
Why : array = np.random.randn(vocab_size, N).astype(val) * rngAmp  # hidden to output

bh : array = np.zeros((4 * N, 1), dtype=val)  # hidden bias
bh[2 * N:3 * N, :] = 1 # i o f c : init f gates biases higher

by = np.zeros((vocab_size, 1), dtype=val)  # output bias

Wrh = np.random.randn(4 * N, MW).astype(val) * rngAmp  # read vector to hidden
Whv = np.random.randn(MW, N).astype(val) * rngAmp  # write content
Whr = np.random.randn(MW, N).astype(val) * rngAmp  # read strength
Whw = np.random.randn(MW, N).astype(val) * rngAmp  # write strength
Whe = np.random.randn(MW, N).astype(val) * rngAmp  # erase strength
Wry = np.random.randn(vocab_size, MR * MW).astype(val) * rngAmp  # erase strength



# re-used.  TODO store in a state class with learn(..) as method
dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
dWhr, dWhv, dWhw, dWhe, dWrh, dWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
mem_erase_gate, mem_new_content, mem_read_gate, mem_write_gate = I * [array], I * [array], I * [array], I * [array]
dmem_write_key, dmem_read_key = key(), key()
for t in range(I):
    mem_write_gate[t], mem_read_gate[t] = gate(), gate()
xs: floatarray = I * [float]
ys: floatarray = I * [float]
ps: floatarray = I * [float]
gs: floatarray = I * [float]
mem_read_key: listOfArrays = I * [array]
mem_write_key: listOfArrays = I * [array]

def learn(cprev, hprev, mprev, rprev, plot=False):
    """
    inputs,targets are both list of integers.
    cprev is HxB array of initial memory cell state
    hprev is HxB array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # TODO remove the need for dictionaries due to negative indexing, so that this can be numpy array
    hs, cs = {}, {}
    memory, rs = {}, {}

    # init previous states
    hs[-1], cs[-1], rs[-1], memory[-1] = np.copy(hprev), np.copy(cprev), np.copy(rprev), np.copy(mprev)


    for t in range(I):
        mem_write_gate[t].fill(0)
        mem_read_gate[t].fill(0)
    dmem_write_key.fill(0)
    dmem_read_key.fill(0)

    loss : float = 0
    # forward pass
    for t in range(I):
        xs[t] = np.zeros((vocab_size, B), dtype=val)  # encode in 1-of-k representation
        for b in range(0, B): xs[t][:, b][inputs[t][b]] = 1

        # gates, linear part + previous read vector
        gsT : array = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh + np.dot(Wrh, rs[t - 1])
        gs[t] = gsT

        # gates nonlinear part
        gsT[0:3 * N, :] = sigmoid(gsT[0:3 * N, :])  # i, o, f gates
        gsT[3 * N:4 * N, :] = np.tanh(gsT[3 * N:4 * N, :])  # c gate

        # mem(t) = c gate * i gate + f gate * mem(t-1)
        cs[t] = gsT[3 * N:4 * N, :] * gsT[0:N, :] + gsT[2 * N:3 * N, :] * cs[t - 1]
        cs[t] = np.tanh(cs[t])  # mem cell - nonlinearity

        hstT: array = gsT[N:2 * N, :] * cs[t]  # new hidden state
        hs[t] = hstT

        ys[t] = np.dot(Why, hstT) + by  # unnormalized log probabilities for next chars

        ##### external mem ########
        mem_read_key[t] = np.dot(Whr, hstT)  # key used for content based read
        mem_write_key[t] = np.dot(Whw, hstT)  # key used for content based read

        mem_new_content[t] = np.dot(Whv, hstT)

        for b in range(0, B):
            # normalize - unit length
            # mem_read_key[t][:,b] = np.exp(mem_read_key[t][:,b]) / np.sum(np.exp(mem_read_key[t][:,b]), axis=0) # probabilities for next chars
            # mem_write_key[t][:,b] = np.exp(mem_write_key[t][:,b]) / np.sum(np.exp(mem_write_key[t][:,b]), axis=0) # probabilities for next chars

            # s = np.sum(np.exp(memory[t-1][:,:,b]), axis=1, keepdims=1)
            # memory[t-1][:,:,b] = np.exp(memory[t-1][:,:,b])/s

            bb: array = memory[t - 1][:, :, b]
            mem_write_gate[t][:,b,None] = np.dot(bb, mem_write_key[t][:,b,None])
            mem_read_gate [t][:,b,None] = np.dot(bb,  mem_read_key[t][:,b,None])

        mem_erase_gate[t] = sigmoid(np.dot(Whe, hstT))

        softmax(mem_write_gate, t)
        softmax(mem_read_gate, t)

        memory[t] = memory[t - 1] * (1 - np.reshape(mem_erase_gate[t], (1, MW, B)) * np.reshape(mem_write_gate[t], (MN, 1, B))) + np.reshape(mem_new_content[t], (1, MW, B)) * np.reshape(mem_write_gate[t], (MN, 1, B))

        rs[t] = np.sum(memory[t] * np.reshape(mem_read_gate[t], (MN, 1, B)), axis=0)

        ys[t] += np.dot(Wry, rs[t]) - np.max(ys[t], axis=0)  # add read vector to output

        psT = np.exp(ys[t] - logsumexp(ys[t], axis=0))  # probabilities for next chars
        # psT = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars
        ps[t] = psT

        for b in range(0, B):
            if psT[targets[t, b], b] > 0: loss += -np.log(psT[targets[t, b], b])  # softmax (cross-entropy loss)

    global dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry
    # backward pass: compute gradients going backwards
    # dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why )
    # dWhr, dWhv, dWhw, dWhe, dWrh, dWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
    for d in [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry]: d.fill(0)

    # TODO mutable operations on re-used arrays:
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dcnext = np.zeros_like(cs[0])
    dhnext = np.zeros_like(hs[0])
    #dmemory = np.zeros_like(memory[0])
    dmem_next = np.zeros_like(memory[0])
    drs_next = np.zeros_like(rs[0])
    dg = np.zeros_like(gs[0])
    W_ones = np.ones((MW, 1), dtype=val)

    for t in reversed(range(I)):
        dy = np.copy(ps[t])
        for b in range(0, B): dy[targets[t][b], b] -= 1  # backprop into y

        hstT: array = hs[t].T

        dWhy += np.dot(dy, hstT)

        ##external######
        dWry += np.dot(dy, rs[t].T)
        drs: float = np.dot(Wry.T, dy) + drs_next

        rgt = mem_read_gate[t]
        dmemory = np.reshape(drs, (1, MW, B)) * np.reshape(rgt, (MN, 1, B)) + dmem_next

        dmem_write_gate = np.dot(W_ones.T, dmemory * (mem_new_content[t] - mem_erase_gate[t] * memory[t - 1]))  # iface gates

        dmem_read_gate = np.dot(W_ones.T, drs * memory[t])

        # propagate back through softmax
        wgt = mem_write_gate[t]
        dmem_write_gate *= wgt
        dmem_write_gate -= wgt * np.sum(dmem_write_gate, axis=1)

        dmem_read_gate *= rgt
        dmem_read_gate -= rgt * np.sum(dmem_read_gate, axis=1)

        dmem_new_content = dmemory * np.reshape(wgt, (MN, 1, B))

        dmem_erase_gate = -dmemory * memory[t - 1] * np.reshape(wgt, (MN, 1, B))
        dmem_next = dmemory * (1 - np.reshape(mem_erase_gate[t], (1, MW, B)) * np.reshape(wgt, (MN, 1, B)))

        for b in range(0, B):
            dmem_next[:, :, b] += np.dot(dmem_read_gate[:, :, b].T, mem_read_key[t][:, b, None].T) + np.dot(dmem_write_gate[:, :, b].T, mem_write_key[t][:, b, None].T)

            #  dmem_next[:,:,b] = dmem_next[:,:,b] * memory[t-1][:,:,b]
            #  dmem_next_sum = np.sum(dmem_next[:,:,b], axis=1, keepdims=1)
            #  dmem_next[:,:,b] -= memory[t-1][:,:,b] * dmem_next_sum

        #  dmem_read_key = dmem_read_gate
        dmem_erase_gate = dmem_erase_gate * mem_erase_gate[t] * (1 - mem_erase_gate[t])

        dmem_write_gate = np.reshape(dmem_write_gate, (MN, B))
        dmem_read_gate = np.reshape(dmem_read_gate, (MN, B))

        for b in range(0, B):
            mt = memory[t - 1][:, :, b].T
            dmem_write_key[:, b, None] = np.dot(mt, dmem_write_gate[:, b, None])
            dmem_read_key[:, b, None] = np.dot(mt, dmem_read_gate[:, b, None])

            #  dmem_read_key[:,b,None] = dmem_read_key[:,b,None] * mem_read_key[t][:,b,None]
            #  dmem_read_key_sum = np.sum(dmem_read_key[:,b,None], axis=0)
            #  dmem_read_key[:,b,None] -= mem_read_key[t][:,b,None] * dmem_read_key_sum

            #  dmem_write_key[:,b,None] = dmem_write_key[:,b,None] * mem_write_key[t][:,b,None]
            #  dmem_write_key_sum = np.sum(dmem_write_key[:,b,None], axis=0)
            #  dmem_write_key[:,b,None] -= mem_write_key[t][:,b,None] * dmem_write_key_sum

        dWhw += np.dot(np.reshape(dmem_write_key, (MW, B)), hstT)
        dWhr += np.dot(np.reshape(dmem_read_key, (MW, B)), hstT)
        dWhe += np.dot(np.sum(dmem_erase_gate, axis=0), hstT)
        dWhv += np.dot(np.sum(dmem_new_content, axis=0), hstT)
        ########

        dby += np.expand_dims(np.sum(dy, axis=1), axis=1)

        dh : float = np.dot(Why.T, dy) + dhnext  # backprop into h
        dh += np.dot(Whw.T, np.reshape(dmem_write_key, (MW, B)))
        dh += np.dot(Whr.T, np.reshape(dmem_read_key,  (MW, B)))
        dh += np.dot(Whe.T, np.sum(dmem_erase_gate, axis=0))
        dh += np.dot(Whv.T, np.sum(dmem_new_content, axis=0))
        # external end ###

        dc = dh * gs[t][N:2 * N, :] + dcnext  # backprop into c
        dc = dc * (1 - cs[t] * cs[t])  # backprop though tanh

        dg[N:2 * N, :] = dh * cs[t]  # o gates
        dg[0:N, :] = gs[t][3 * N:4 * N, :] * dc  # i gates
        dg[2 * N:3 * N, :] = cs[t - 1] * dc  # f gates
        dg[3 * N:4 * N, :] = gs[t][0:N, :] * dc  # c gates

        gst03 = gs[t][0:3 * N, :]
        dg[0:3 * N, :] = dg[0:3 * N, :] * gst03 * (1 - gst03)  # backprop through sigmoids

        gst34 = gs[t][3 * N:4 * N, :]
        dg[3 * N:4 * N, :] = dg[3 * N:4 * N, :] * (1 - gst34 * gst34)  # backprop through tanh

        dbh += np.expand_dims(np.sum(dg, axis=1), axis=1)
        dWxh += np.dot(dg, xs[t].T)
        dWhh += np.dot(dg, hs[t - 1].T)
        dWrh += np.dot(dg, rs[t - 1].T)
        dhnext : float = np.dot(Whh.T, dg)
        drs_next : float = np.dot(Wrh.T, dg)
        dcnext = dc * gs[t][2 * N:3 * N, :]

    if plotting and plot:
        _b_ = 1  # sequence number
        plt.subplot(2, 1, 1)
        plt.imshow(memory[MW][:, :, _b_])
        plt.colorbar()
        plt.title('memory state')
        plt.subplot(2, 1, 2)
        read_gates_history = np.zeros((S, MN), dtype=val)
        for i in range(0, S): read_gates_history[i, :] = mem_read_gate[i][:, _b_]
        plt.imshow(read_gates_history.T)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('activation', rotation=270)
        plt.ylabel('location')
        plt.yticks(np.arange(0, MN))
        plt.xlabel('time step')
        plt.title('mem read gate in time')
        plt.show()

    if clipGradients:
        for dparam in [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dbh, dby, cs[I - 1], hs[I - 1]


def softmax(gate:array, t:int):
    gate[t] = np.exp(gate[t])
    mem_read_gate_sum = np.sum(gate[t], axis=0)
    gate[t] /= mem_read_gate_sum


def sample(c, h, m, r, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1), dtype=val)
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        g = np.dot(Wxh, x) + np.dot(Whh, h) + np.dot(Wrh, r) + bh
        g[0:3 * N, :] = sigmoid(g[0:3 * N, :])
        g[3 * N:4 * N, :] = np.tanh(g[3 * N:4 * N, :])
        c = g[3 * N:4 * N, :] * g[0:N, :] + g[2 * N:3 * N, :] * c
        c = np.tanh(c)
        h = g[N:2 * N, :] * c
        y = np.dot(Why, h) + by
        mem_new_content = np.dot(Whv, h)
        # mem_write_gate = sigmoid(np.dot(Whw, h))
        # mem_read_gate = sigmoid(np.dot(Whr, h))
        mem_write_key = np.dot(Whw, h)
        mem_read_key = np.dot(Whr, h)
        mem_write_gate = np.exp(mem_write_key)
        mem_read_gate = np.exp(mem_read_key)
        mem_write_gate_sum = np.sum(mem_write_gate, axis=0)
        mem_read_gate_sum = np.sum(mem_read_gate, axis=0)
        mem_write_gate = mem_write_gate / mem_write_gate_sum
        mem_read_gate = mem_read_gate / mem_read_gate_sum
        mem_erase_gate = sigmoid(np.dot(Whe, h))
        # print m.shape
        m = m * (1 - np.reshape(mem_erase_gate, (1, MW)) * np.reshape(mem_write_gate, (MN, 1)))  # 1
        #  m = m * (1-mem_erase_gate) # 2
        m += np.reshape(mem_new_content, (1, MW)) * np.reshape(mem_write_gate, (MN, 1))
        r = np.expand_dims(np.sum(m * np.reshape(mem_read_gate, (MN, 1)), axis=0), axis=1)
        y += np.dot(Wry, r)
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1), dtype=val)
        x[ix] = 1
        ixes.append(ix)
    return ixes




v = 0
n = 0
p = np.random.randint(len(data) - 1 - S, size=B).tolist()

targets : array = np.zeros((S, B), dtype=int) #maybe short int
cprev = np.zeros((N, B), dtype=val)
hprev = np.zeros((N, B), dtype=val)
mprev = np.zeros((MN, MW, B), dtype=val)
rprev = np.zeros((MW, B), dtype=val)
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mWhr, mWhv, mWhw, mWhe, mWrh, mWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(
    Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * S  # loss at iteration 0
start = time.time()

t = 0
last = start
while t < T:
    # prepare inputs (we're sweeping from left to right in steps S long)
    for b in range(0, B):
        if p[b] + S + 1 >= len(data) or n == 0:
            cprev[:, b] = np.zeros(N, dtype=val)  # reset LSTM memory
            hprev[:, b] = np.zeros(N, dtype=val)  # reset hidden memory
            mprev[:, :, b] = np.random.randn(MN, MW) * rngAmp  # reset ext memory
            rprev[:, b] = np.zeros(MW, dtype=val)  # reset read vec memory
            p[b] = np.random.randint(len(data) - 1 - S)

        Pb = p[b]
        inputs[:, b] = [char_to_ix[ch] for ch in data[Pb:Pb + S]]
        targets[:, b] = [char_to_ix[ch] for ch in data[Pb + 1:Pb + S + 1]]

    # sample from the model now and then
    if (n + 1) % opt.report_interval == 0:
        # if log..
        sample_ix = sample(np.expand_dims(cprev[:, 0], axis=1), np.expand_dims(hprev[:, 0], axis=1),
                           (mprev[:, :, 0]), np.expand_dims(rprev[:, 0], axis=1), inputs[0], opt.sample_length)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))
        entry = '%s\n' % txt
        with open(samplelogname, "w") as myfile:
            myfile.write(entry)

        if GC:
            gradCheck(targets, cprev, hprev, mprev, rprev)

    plot = n % opt.report_interval == 200 and n > 0

    # forward S characters through the net and fetch gradient
    loss, dbh, dby, cprev, hprev = learn(cprev, hprev, mprev, rprev, plot)
    smooth_loss = smooth_loss * 0.999 + np.mean(loss) / (np.log(2) * B) * 0.001
    interval = time.time() - last

    if n % opt.report_interval == 0 and n > 0:
        # if log..
        tdelta = time.time() - last
        last = time.time()
        t = time.time() - start
        entry = '{:5}\t\t{:3f}\t{:3f}\n'.format(n, t, smooth_loss / S)
        with open(logname, "a") as myfile: myfile.write(entry)

        print('%2d: %.3f s, iter %d, %.4f BPC, %.2f char/s' % (
        v, t, n, smooth_loss / S, (B * S * 100) / tdelta))  # print progress

    COMPONENTS = [(Wxh, dWxh, mWxh), (Whh, dWhh, mWhh), (Why, dWhy, mWhy), (Whr, dWhr, mWhr), (Whv, dWhv, mWhv),
                  (Whe, dWhe, mWhe), (Wrh, dWrh, mWrh), (Wry, dWry, mWry), (bh, dbh, mbh), (by, dby, mby)]
    for param, dparam, mem in COMPONENTS:
        # perform parameter update with Adagrad
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    for b in range(0, B): p[b] += S  # move data pointer
    n += 1  # iteration counter

# procs = [Process(target=proc, args=(i,)) for i in range(0,8)]
# for p in procs:
#    p.start()
#    time.sleep(0.5)
# for p in procs: p.join()
