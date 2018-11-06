import glob
import pickle
import numpy as np
from keras.preprocessing import text, sequence

base_path = 'C:/Users/99263/Data/rjkg/'
train_path = 'ruijin_round1_train2_20181022/'
test_path = 'ruijin_round1_test_a_20181022/'

train_text = glob.glob(base_path + train_path + '*.txt')
train_ner = [i.replace('.txt', '.ann') for i in train_text]
#  train_ner = glob.glob(base_path + train_path + '*.ann')
test_text = glob.glob(base_path + test_path + '*.txt')
with open('test_text.pick', 'wb')as f:
    pickle.dump(test_text, f)


def pre_1():
    tag = set()
    anns = list()
    texts = list()
    chars = set()
    ann_seq = list()

    texts_t = list()

    for annf in train_ner:
        with open(annf, 'r', encoding='utf-8') as f:
            ann = [i.strip('\n').split('\t') for i in f.readlines()]
            for i in ann:
                i.insert(2, i[1].replace(';', ' ').split()[1:])
                i[1] = i[1].split()[0]
                tag.add(i[1])
        anns.append(ann)
    length = []
    for txt in train_text:
        with open(txt, 'r', encoding='utf-8') as f:
            t = f.read()
            # rt = t.replace('。', ',').split(',')
            texts.append(t)
            chars.update(list(t))
            length.append(len(t))
            # length.append(max([len(i) for i in rt]))
    max_length = max(length)
    avg = sum(length) / len(length)
    print(max_length, avg)

    for txt in test_text:
        with open(txt, 'r', encoding='utf-8') as f:
            t = f.read()
            texts_t.append(t)
            chars.update(list(t))

    chars = list(chars)
    chars.insert(0, '</pad>')
    chars.insert(1, '</not in dict>')
    print(len(chars))
    chars = dict(zip(chars, [i for i in range(len(chars))]))
    ntag = ['<pad/>', '<not-entity/>']
    tag = list(tag)
    for i in tag:
        i0 = i + '-s'
        i1 = i + '-m'
        i2 = i + '-e'
        ntag.extend([i0, i1, i2])

    print(len(ntag))
    tag = dict(zip(ntag, [i for i in range(len(ntag))]))

    # with open('char_list.pick', 'wb') as f:
    #     pickle.dump(chars, f)
    # with open('texts.pick', 'wb') as f:
    #     pickle.dump(texts, f)
    # with open('texts_t.pick', 'wb') as f:
    #     pickle.dump(texts_t, f)
    # with open('tag.pick', 'wb') as f:
    #     pickle.dump(tag, f)
    # with open('anns.pick', 'wb') as f:
    #     pickle.dump(anns, f)

    token = text.Tokenizer()
    token.fit_on_texts(chars)

    text_seq = []
    for i in texts:
        one = []
        for ii in i:
            one.append(int(chars.get(ii)))
        # one = np.array(one).astype('int16')
        text_seq.append(one)

    text_seq_t = []
    for i in texts:
        one = []
        for ii in i:
            one.append(int(chars.get(ii)))
        # one = np.array(one).astype('int16')
        text_seq_t.append(one)

    # text_seq = token.texts_to_sequences(texts)
    # text_seq = sequence.pad_sequences(text_seq, maxlen=max_length, padding='post', truncating='post')
    for i in range(len(anns)):
        tmp = np.ones(len(text_seq[i]), dtype='int16')
        tmp = tmp.tolist()

        for ii in anns[i]:
            # a = texts[0][int(ii[2][0])]
            # b = texts[0][int(ii[2][-1])]
            tmp[int(ii[2][0])] = tag[ii[1] + '-s']
            for iii in range(int(ii[2][0]) + 1, int(ii[2][-1])):
                tmp[iii] = tag[ii[1] + '-m']
            tmp[int(ii[2][-1]) - 1] = tag[ii[1] + '-e']
        ann_seq.append(tmp)

    # text_seq = np.array(text_seq).astype('int16')
    # ann_seq = np.array(ann_seq).astype('int16')

    # with open('chars.pick', 'wb') as f:
    #     pickle.dump(chars, f)
    # with open('tsq.pick', 'wb') as f:
    #     pickle.dump(text_seq, f)
    # with open('tsqt.pick', 'wb') as f:
    #     pickle.dump(text_seq_t, f)
    # with open('ann_seq.pick', 'wb') as f:
    #     pickle.dump(ann_seq, f)

    with open('tag.pick', 'rb') as f:
        tag = pickle.load(f)
    tag = dict(zip(tag.values(), tag.keys()))
    validation_num = 50

    allen = ''
    for i in range(len(texts) - validation_num):
        new = [tag.get(ann_seq[i][ii]) for ii in range(len(texts[i]))]
        new2 = ''
        for ii in range(len(texts[i])):
            new2 += '%s###%s ' % (texts[i][ii], new[ii])
        allen += new2 + '\n'
    with open('allen.txt', 'w', encoding='utf-8') as f:
        f.write(allen)

    allen = ''
    for i in range(len(texts) - validation_num, len(texts)):
        new = [tag.get(ann_seq[i][ii]) for ii in range(len(texts[i]))]
        new2 = ''
        for ii in range(len(texts[i])):
            new2 += '%s###%s ' % (texts[i][ii], new[ii])
        allen += new2 + '\n'
    with open('allen_val.txt', 'w', encoding='utf-8') as f:
        f.write(allen)


def text_():
    for i in train_text:
        with open(i, 'r', encoding='utf-8') as f:
            t = f.read()
            sentence = t.split('。')
            sentence_pair = [i.split(',') for i in sentence]
            for ii in sentence_pair:
                for iii in range(len(ii)):
                    ii[iii] = list(ii[iii])
            yield sentence_pair


if __name__ == '__main__':
    pre_1()
    # with open('texts.pick', 'rb') as f:
    #     texts = pickle.load(f)
    # with open('anns.pick', 'rb') as f:
    #     anns = pickle.load(f)
    # with open('tag.pick', 'rb') as f:
    #     tag = pickle.load(f)
    print(1)
