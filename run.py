import pickle
import random
import numpy as np
from model import model

with open('tsq.pick', 'rb') as f:
    x = pickle.load(f)
with open('ann_seq.pick', 'rb') as f:
    y = pickle.load(f)


def dgen(batch_size=32):
    while 1:
        a, b, c = [], [], []
        for i in range(0, batch_size):
            ii = random.randint(0, len(x[:-50]))
            a.append(x[ii])
            b.append(y[ii])
        a = np.array(a)
        b = np.array(b)
        yield a, b


def dgen_v(batch_size=32):
    while 1:
        a, b, c = [], [], []
        for i in range(0, batch_size):
            ii = random.randint(0, len(x[-50:]))
            a.append(x[ii])
            b.append(y[ii])
        a = np.array(a)
        b = np.array(b)
        yield a, b


m = model()
m.fit_generator(dgen(), steps_per_epoch=100, epochs=30, validation_data=dgen_v(), validation_steps=20)
m.save_weights('1.h5')
