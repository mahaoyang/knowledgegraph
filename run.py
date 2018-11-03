import pickle
import random
import numpy as np
from model import model, units

with open('tsq.pick', 'rb') as f:
    x = pickle.load(f)
with open('ann_seq.pick', 'rb') as f:
    y = pickle.load(f)


def dgen(batch_size=32):
    while 1:
        a, b, c = [], [], []
        for i in range(0, batch_size):
            ii = random.randint(0, len(x[:-50]))
            if len(x[ii]) < units:
                for index in range(len(x[ii]), units):
                    x[ii].append(0)
                    y[ii].append(0)
            iii = random.randint(0, len(x[ii]) - units)
            a.append(x[ii][iii:iii + units])
            b.append(y[ii][iii:iii + units])
        a = np.array(a).astype('int8')
        b = np.array(b).astype('int8')
        b = np.expand_dims(b, 2)
        yield a, b


def dgen_v(batch_size=32):
    while 1:
        a, b, c = [], [], []
        for i in range(0, batch_size):
            ii = random.randint(0, len(x[-50:]))
            if len(x[ii]) < units:
                for index in range(len(x[ii]), units):
                    x[ii].append(0)
                    y[ii].append(0)
            iii = random.randint(0, len(x[ii]) - units)
            a.append(x[ii][iii:iii + units])
            b.append(y[ii][iii:iii + units])
        a = np.array(a).astype('int8')
        b = np.array(b).astype('int8')
        b = np.expand_dims(b, 2)
        yield a, b


batch_size = 64
m = model()
# m.load_weights('1.h5')
m.fit_generator(dgen(batch_size), steps_per_epoch=100, epochs=3, validation_data=dgen_v(batch_size), validation_steps=20)
m.save_weights('1.h5')
