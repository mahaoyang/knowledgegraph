import glob

base_path = 'C:/Users/99263/Data/rjkg/'
train_path = 'ruijin_round1_train2_20181022/'

train_text = glob.glob(base_path + train_path + '*.txt')


def text():
    for i in train_text:
        with open(i, 'r', encoding='utf-8') as f:
            yield f.read()


if __name__ == '__main__':
    for i in text():
        print(i)
