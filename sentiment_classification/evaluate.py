import math
import os
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from Dataset import DataIterator
from tensorboardX import SummaryWriter
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import logging


# 初始化设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始数据集
data = DataIterator.from_corpus(
    corpus_tag_or_dir='./data/weibo_senti_100k',
    tokenizer_path='./chinese-roberta-wwm-ext',
    batch_size=32,
    shuffle=False, # DataItrator的shuffle需要False
    device=device
)
# 获得训练集迭代器、验证集迭代器、配置config参数
train_iter, eval_iter, config = data['train_iter'], data['eval_iter'], data['config']
total_eval_steps = math.ceil(config['eval_size'] * 1 / config['batch_size'])
# 加载模型
model = torch.load('./sentiment_classification/output/pytorch_model_3188.h5', map_location=device)
model.eval()
# 将模型拷贝到设备上
model.to(device)
# 保存预测结果
predictions = []
# 保存实际标签
labels = []
print('evaluate...')
print('total_eval_steps:', total_eval_steps)
for eval_step in range(total_eval_steps):
    print('now step: ', eval_step)
    # 通过next得到one_batch数据
    eval_data = next(eval_iter)
    # 提取真实label标签
    batch_labels = eval_data['labels']
    # 将真实标签加入到labels中
    labels += batch_labels.detach().cpu().numpy().tolist()
    # 前向forward
    eval_output = model(**eval_data)
    # 通过argmax提取概率最大值的索引来获得预测标签的id
    batch_predictions = torch.argmax(eval_output.logits, dim=-1).detach().cpu().numpy().tolist()
    # 将预测结果加入到predictions
    predictions += batch_predictions
# 计算weighted的f1_score，超过二分类的任务，通常会使用weighted的f1进行评测
f1 = f1_score(labels, predictions, average='weighted')
print('f1:', f1)
