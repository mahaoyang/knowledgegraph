import pickle
import random
import numpy as np
from model import model, units, tag_num, MAX_WORD_INDEX

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
                    x[ii].append(1)
                    y[ii].append(0)
            iii = random.randint(0, len(x[ii]) - units)
            a.append(x[ii][iii:iii + units])
            b.append(y[ii][iii:iii + units])
        a = np.array(a).astype('int16')
        b = np.array(b).astype('int16')
        b = np.expand_dims(b, 2)
        yield a, b


def dgen_v(batch_size=32):
    while 1:
        a, b, c = [], [], []
        for i in range(0, batch_size):
            ii = random.randint(0, len(x[-50:]))
            if len(x[ii]) < units:
                for index in range(len(x[ii]), units):
                    x[ii].append(1)
                    y[ii].append(0)
            iii = random.randint(0, len(x[ii]) - units)
            a.append(x[ii][iii:iii + units])
            b.append(y[ii][iii:iii + units])
        a = np.array(a).astype('int16')
        b = np.array(b).astype('int16')
        b = np.expand_dims(b, 2)
        yield a, b


def train():
    batch_size = 8
    m = model()
    m.load_weights('1.h5')
    m.fit_generator(dgen(batch_size), steps_per_epoch=200, epochs=1600, validation_data=dgen_v(batch_size),
                    validation_steps=10)
    m.save_weights('1.h5')


def predict():
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
    pxn = np.array(pxn).astype('int16')
    m = model()
    m.load_weights('1.h5')
    pred = m.predict(pxn)
    tag_p = []
    for i in pred:
        tag_idx = np.argmax(i, 1)
        # tagg = []
        # for ii in tag_idx:
        #     tagg.append(tag.get(ii))
        tag_p.append(tag_idx)
    tag_texts = []
    for i in range(len(length) - 1):
        text_one = []
        for ii in tag_p[length[i]:length[i + 1]]:
            text_one.extend(ii)
        tag_texts.append(text_one)
    with open('texts_pred_tag.pick', 'wb') as f:
        pickle.dump(tag_texts, f)


import sys
import time

sys.setrecursionlimit(10000000)


def submit():
    with open('tag.pick', 'rb') as f:
        tag = pickle.load(f)
    tag = dict(zip(tag.values(), tag.keys()))
    with open('texts_pred_tag.pick', 'rb') as f:
        tag_texts = pickle.load(f)

    def cut(i, tg_one, posi, s=0):
        for ii in range(s, len(i)):
            if i[ii] != 1 and i[ii] != 0:

                for iii in range(ii, len(i)):
                    if i[iii] == i[ii] + 2 or i[iii] == i[ii] + 1:
                        if i[iii] == i[ii] + 2:
                            posi.append([ii, iii + 1])
                            tg_one.append(tag.get(i[ii]).split('-')[0])
                            if iii + 1 < len(i):
                                print('1', time.time())
                                return cut(i, tg_one, posi, s=iii + 1)
                else:
                    break

    tgd = []
    posid = []
    for i in tag_texts:
        tg_one = []
        posi = []
        cut(i, tg_one, posi)
        tgd.append(tg_one)
        posid.append(posi)

    with open('texts_t.pick', 'rb') as f:
        texts_t = pickle.load(f)

    with open('test_text.pick', 'rb')as f:
        test_text = pickle.load(f)

    def subb(i, subm, s=0):
        ttp = texts_t[i]
        tgp = tgd[i]
        pot = posid[i]
        print('length', len(tgp), len(pot))
        sstr = 'T%s\t%s %s %s\t%s\n'
        ssstr = 'T%s\t%s %s %s;%s %s\t%s %s\n'
        for ii in range(s, len(tgp)):
            if ii + 1 < len(tgp):
                print('2', time.time())
                print(pot[ii])
                print(pot[ii + 1])
                print(tgp[ii])
                print(tgp[ii + 1])
                if pot[ii][1] + 1 == pot[ii + 1][0] and tgp[ii] == tgp[ii + 1]:
                    st = ssstr % (
                        ii, tgp[ii], pot[ii][0], pot[ii][1], pot[ii + 1][0], pot[ii + 1][1],
                        ttp[pot[ii][0]:pot[ii][1]], ttp[pot[ii + 1][0]:pot[ii + 1][1]])
                    subm += st
                    return subb(i, subm, ii + 2)
                else:
                    st = sstr % (ii, tgp[ii], pot[ii][0], pot[ii][1], ttp[pot[ii][0]:pot[ii][1]])
                    subm += st
        return subm

    for i in range(len(texts_t)):
        subm = ''
        subm = subb(i, subm)
        ttt = test_text[i].replace('\\', '/').split('/')[-1].split('.')[0]
        print('subm', subm)
        fnm = 'C:/Users/99263/Data/rjkg/submit/%s.ann' % ttt
        with open(fnm, 'w', encoding='utf-8') as f:
            f.write(subm)


if __name__ == '__main__':
    # with open('C:/Users/99263/Data/rjkg/ruijin_round1_submit_20181022/test_id.ann', 'r') as f:
    #     submit_test = f.read()
    #     b = submit_test.split('\t')
    #     print(submit_test)
    train()
    predict()
    submit()
    # with open('C:/Users/99263/Data/rjkg/ruijin_round1_submit_20181022/%s.ann' % 123, 'w') as f:
    #     a = '123'
    #     submit_test = f.write(a)
