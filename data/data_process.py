import json

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split


def get_stopwords(stop_words_path):
    '''加载停用词'''
    stop_words = []
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            stop_words.append(line.strip())
    return stop_words


def save_data(reviews, labels, save_path):
    # 将数据保存为json格式
    with open(save_path, 'w', encoding='utf8') as f:
        for review, label in zip(reviews, labels):
            dic = {}
            dic['inputs'] = review
            dic['labels'] = '正面' if int(label) == 1 else '负面'
            f.write(json.dumps(dic, ensure_ascii=False))
            f.write('\n')


def data_process(weibo_path):
    df = pd.read_csv(weibo_path, encoding='utf8')
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.15, random_state=0)
    print('X_train:', len(X_train), 'X_test:', len(X_test))
    save_data(X_train, y_train, './data/weibo_senti_100k/train.json')
    save_data(X_test, y_test, './data/weibo_senti_100k/eval.json')
    # 保存config文件
    config_dic = {}
    config_dic['train_size'] = len(X_train)
    config_dic['eval_size'] = len(X_test)
    config_dic['task_tag'] = 'sentiment_classification'
    config_dic['labels_map'] = {'负面': 0, '正面': 1}
    config_dic['labels_balance'] = {
        'train_labels': {'负面': len(y_train) - sum(y_train), '正面': sum(y_train)},
        'eval_labels': {'负面': len(y_test) - sum(y_test), '正面': sum(y_test)},
    }
    with open('./data/weibo_senti_100k/config.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(config_dic, ensure_ascii=False))
    print('finish!')


if __name__ == '__main__':
    weibo_path = './data/weibo_senti_100k/weibo_senti_100k.csv'
    data_process(weibo_path)