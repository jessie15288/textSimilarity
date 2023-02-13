import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
import json

num = 100  # 采样数
topk = 5  # 返回 topk 个结果

# 读取数据集
sentences = []
fname = "/Users/jyang/datasets/nlp/text_similarity/oppo/oppp.json"

with open(fname, 'r', encoding="utf-8") as f:
    data = json.load(f)

    # data结构里面包含【train, test, dev】
    # 其中train（17万条）和dev（1万条）包含句子对和label，而test（5万条）只包含句子对。label里1表示句子是同义的，0表示非同义
    for i, sample in enumerate(data["test"]):
        if i < 100:
            sentences.append(sample["q1"])
print('%d questions loaded, avg.len %d' % (len(sentences), np.mean([len(d) for d in sentences])))

with BertClient(port=5555, port_out=5556) as bc:
    # 获取句子向量编码
    doc_vecs = bc.encode(sentences)

    while True:
        query = input(colored('your question：', 'green'))
        query_vec = bc.encode([query])[0]

        # 余弦相似度 分数计算。
        # np.linalg.norm 是求取向量的二范数，也就是向量长度。
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)

        '''
        		argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)

            [::-1]取从后向前（相反）的元素, 例如[ 1 2 3 4 5 ]
            则输出为[ 5 4 3 2 1 ]
        '''
        topk_idx = np.argsort(score)[::-1][:topk]
        print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
        for idx in topk_idx:
            print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(sentences[idx], 'yellow')))