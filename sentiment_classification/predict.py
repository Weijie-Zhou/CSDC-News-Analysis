import math
import time
import json
import os
import logging
import pandas as pd
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from Dataset import DataIterator
from tensorboardX import SummaryWriter
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import f1_score


# 初始化设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('./chinese-roberta-wwm-ext')
# 加载模型
model = torch.load('./sentiment_classification/output/pytorch_model.h5', map_location=device)
model.eval()
# 将模型拷贝到设备上
model.to(device)
# 加载labels_map
labels_map = {0: '负面', 1: "正面"}

def predict_one(review):
    if type(review) == str:
        review = [review]
    encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=review,
                                          padding=True,
                                          max_length=128,
                                          truncation=True)
    new_batch = {}
    for key, value in encoded.items():
        new_batch[key] = torch.tensor(value)
    eval_output = model(**new_batch)
    # 通过argmax提取概率最大值的索引来获得预测标签的id
    batch_predictions = torch.argmax(eval_output.logits, dim=-1).detach().cpu().numpy().tolist()
    # 将预测结果加入到predictions
    return labels_map[batch_predictions[0]]

def predict_batch(reviews, batch_size):
    predict_steps = math.ceil(len(reviews) / batch_size)
    print('predict_steps: ', predict_steps)
    # 保存预测结果
    predictions = []
    for i in range(3858, predict_steps):
        print('now step: ', i)
        review_lst = list(reviews[i*batch_size: (i+1)*batch_size])
        encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=review_lst,
                                                padding=True,
                                                max_length=128,
                                                truncation=True)
        new_batch = {}
        for key, value in encoded.items():
            new_batch[key] = torch.tensor(value)
        eval_output = model(**new_batch)
        # 通过argmax提取概率最大值的索引来获得预测标签的id
        batch_predictions = torch.argmax(eval_output.logits, dim=-1).detach().cpu().numpy().tolist()
        # 将预测结果加入到predictions
        predictions += [labels_map[prediction] for prediction in batch_predictions]
    return predictions



if __name__ == '__main__':
    start = time.time()
    comment_data_yiqing_limit = pd.read_csv('./news/preprocess_data/疫情新闻评论_筛选.csv')
    print(len(comment_data_yiqing_limit))
    comment_data_yiqing_limit['情感倾向'] = predict_batch(comment_data_yiqing_limit['评论内容'], 32)
    print(time.time() - start)
    comment_data_yiqing_limit.to_csv('疫情新闻评论_筛选_情感倾向.csv')