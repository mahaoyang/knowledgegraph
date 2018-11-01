import glob
import pickle
import numpy as np
from keras.preprocessing import text, sequence

base_path = 'C:/Users/99263/Data/rjkg/'
train_path = 'ruijin_round1_train2_20181022/'

train_text = glob.glob(base_path + train_path + '*.txt')
train_ner = glob.glob(base_path + train_path + '*.ann')


def pre_1():
    tag = set()
    anns = list()
    texts = list()
    chars = set()
    ann_seq = list()
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

    chars = list(chars)
    print(len(chars))
    # chars = dict(zip(chars, [i for i in range(len(chars))]))
    tag = list(tag)
    tag = dict(zip(tag, [i for i in range(len(tag))]))

    with open('char_list.pick', 'wb') as f:
        pickle.dump(chars, f)
    with open('texts.pick', 'wb') as f:
        pickle.dump(texts, f)
    with open('tag.pick', 'wb') as f:
        pickle.dump(tag, f)
    with open('anns.pick', 'wb') as f:
        pickle.dump(anns, f)

    token = text.Tokenizer()
    token.fit_on_texts(chars)
    text_seq = sequence.pad_sequences(token.texts_to_sequences(texts), maxlen=max_length, padding='post',
                                      truncating='post')
    for i in range(len(anns)):
        tmp = np.zeros((max_length,), dtype='int16')

        for ii in anns[i]:
            tag_n = tag[ii[1]] + 1
            tmp[int(ii[2][0])] = tag_n * 3
            tmp[int(ii[2][0]) + 1:int(ii[2][-1])] = tag_n * 3 + 1
            tmp[int(ii[2][-1]) - 1] = tag_n * 3 + 2
        ann_seq.append(tmp)

    text_seq = np.array(text_seq).astype('int16')
    ann_seq = np.array(ann_seq).astype('int16')

    with open('token.pick', 'wb') as f:
        pickle.dump(token, f)
    with open('tsq.pick', 'wb') as f:
        pickle.dump(text_seq, f)
    with open('ann_seq.pick', 'wb') as f:
        pickle.dump(ann_seq, f)


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
    with open('texts.pick', 'rb') as f:
        texts = pickle.load(f)
    with open('anns.pick', 'rb') as f:
        anns = pickle.load(f)
    with open('tag.pick', 'rb') as f:
        tag = pickle.load(f)
    print()
