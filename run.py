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


def train():
    batch_size = 8
    m = model()
    m.load_weights('1.h5')
    m.fit_generator(dgen(batch_size), steps_per_epoch=300, epochs=10, validation_data=dgen_v(batch_size),
                    validation_steps=20)
    m.save_weights('1.h5')


def predict():
    with open('tag.pick', 'rb') as f:
        tag = pickle.load(f)
    tag = dict(zip(tag.values(), tag.keys()))
    px = []
    with open('tsqt.pick', 'rb') as f:
        x = pickle.load(f)
    for i in range(len(x)):
        pxt = []
        for index in range(0, len(x[i]), units):
            xp = x[i][index:index + units]
            for ii in range(len(xp), units):
                xp.append(0)
            pxt.append(xp)
        px.append(pxt)
    length = [0]
    pxn = []
    for i in range(len(px)):
        length.append(length[i - 1] + len(px[i]))
        pxn.extend(px[i])
    pxn = np.array(pxn[0:2]).astype('int8')
    m = model()
    m.load_weights('1.h5')
    pred = m.predict(pxn)
    tag_p = []
    for i in pred:
        tag_idx = np.argmax(i, 1)
        tagg = []
        for ii in tag_idx:
            if tag.get(ii) == None:
                tagg.append('</not in dict>')
            else:
                tagg.append(tag.get(ii))
        tag_p.append(tagg)
    return tag_p


if __name__ == '__main__':
    # train()
    predict()
