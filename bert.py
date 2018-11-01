import keras
from keras_bert import get_base_dict, get_model, gen_batch_inputs

from data import text_

sen_n = []


def stp():
    for i in text_():
        for ii in i:
            for iii in ii:
                sen = []
                for iiii in range(0, len(iii), 18):
                    if len(iii) - iiii < 18:
                        sen.append(iii[-18:])
                    else:
                        sen.append(iii[iiii: iiii + 18])
                sen_n.append(sen)

    sen = []
    for i in sen_n:
        for ii in range(len(i) - 1):
            sen.append([i[ii], i[ii + 1]])
    for i in sen:
        yield i


sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]
sentence_pairs = [i for i in stp()]
token_dict = get_base_dict()
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.1
)
model.summary()


def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )


model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)
