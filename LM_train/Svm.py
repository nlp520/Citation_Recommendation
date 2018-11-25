#!usr/bin/python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
import pickle
import random
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import jieba

from process import jsonload, pickleload


def tokenize(s):
    return s.split(' ')
'''
12gram：0.3713539399216369
12gram + 句子个数信息 + 词的个数信息：准确率为：0.378
12gram + 句子个数信息 + 词的个数信息：准确率为: 0.41796875
全部特征准确率为: 0.63671875
全部特征：准确率为: 0.640625
12gram:准确率为: 0.64453125

训练集：（全部特征）
准确率为: 0.7884196778406617
准确率为: 0.7892903787548977
'''

def getTfidf():
    '''
    up_source.append(data['up_source'])
        down_source.append(data['down_source'])
        target.append(data['target'])
        cit_up_source.append(data['cit_up_source'])
        cit_down_source.append(data['cit_down_source'])
        cit_target.append(data['cit_target'])
        score.append(data['bleu1_score'])
    :return:
    数量：189389
    '''
    train_data = pickleload('../data/train_data.pkl', 'train_data.pkl' )
    sentences = []
    for data in tqdm(train_data):
        up_source = data['up_source_tokens']
        down_source = data['down_source_tokens']
        target = data['target_tokens']
        if up_source != "":
            sentences.append(up_source)
        if down_source != "":
            sentences.append(down_source)
        if target != "":
            sentences.append(target)
    #创建词向量空间
    vec = TfidfVectorizer(ngram_range=(1, 2),
                          min_df=3, max_df=0.9, #strip_accents='unicode',
                          use_idf=1, smooth_idf=1, sublinear_tf=1)
    print('开始训练tfidf')
    vec.fit_transform(sentences)
    pickle.dump(vec, open('./svmModelsave/tfidf_12gram.pkl','wb'))#25453
    dic = vec.vocabulary_
    print(len(dic))

def train_svm():
    '''
    训练分类器
    :return:
    '''
    random.seed(5)
    getTfidf()
    datas = pd.read_csv('./train_data/train_data.csv')
    content_sources = datas['content_source'].values[0:len(datas)*4//5]  # 字级别   对应  word
    cit_content_sources = datas['cit_content_source'].values[0:len(datas)*4//5]  # 字级别   对应  word
    targets = datas['target'].values[0:len(datas)*4//5]  # 字级别   对应  word
    cit_targets = datas['cit_target'].values[0:len(datas)*4//5]  # 字级别   对应  word

    labels = datas['score'].values

    train_data, dev_data, train_label, dev_label= train_test_split(data, labels, test_size=0.1, random_state=5)
    print('加载tfidf模型')
    vec = pickle.load(open('./model/tfidf_12gram.pkl', 'rb'))
    print('转化训练集')
    train = vec.transform(train_data)
    dev = vec.transform(dev_data)

    train_word_sizes, train_sen_sizes, train_highword_sizes, train_wc_sizes, train_im_sizes, train_conj_sizes, train_yinhao_sizes, train_shuminghao_sizes= getFeature(train_data, feature)
    #
    dev_word_sizes, dev_sen_sizes,dev_highword_sizes, dev_wc_sizes, dev_im_sizes, dev_conj_sizes, dev_yinhao_sizes, dev_shuminghao_sizes= getFeature(dev_data, feature)
    train = csr_matrix(train).toarray()
    dev = csr_matrix(dev).toarray()
    # print(train.shape)
    # print(train_word_sizes.shape)
    # print(train_sen_sizes.shape)
    # print(train_highword_sizes.shape)
    # print(train_wc_sizes.shape)
    # print(train_im_sizes.shape)
    # print(train_conj_sizes.shape)
    # train = np.hstack((train, train_word_sizes, train_sen_sizes, train_highword_sizes, train_wc_sizes, train_im_sizes, train_conj_sizes,train_yinhao_sizes,train_shuminghao_sizes))
    # dev = np.hstack((dev, dev_word_sizes, dev_sen_sizes,dev_highword_sizes, dev_wc_sizes, dev_im_sizes, dev_conj_sizes,dev_yinhao_sizes, dev_shuminghao_sizes))
    train = np.hstack((train, train_highword_sizes, train_wc_sizes, train_im_sizes,train_yinhao_sizes, train_sen_sizes, train_conj_sizes))
    dev = np.hstack((dev, dev_highword_sizes, dev_wc_sizes,dev_im_sizes,dev_yinhao_sizes ,dev_sen_sizes, dev_conj_sizes))
    lr = LogisticRegression() #使用逻辑斯特回归进行预测
    # lr = LinearSVR()#使用svm进行预测
    print('开始训练回归模型')
    lr.fit(train, train_label)
    print("训练完回归模型，开始保存模型")
    pickle.dump(lr, open('./model/classifyModel1.pkl', 'wb'))#classifyModel0
    print("训练完回归模型，开始进行预测")
    predict = lr.predict(dev)
    print("预测结果为：")
    count = 0
    for i in range(len(predict)):
        flag = testError(predict[i], dev_label[i])
        if flag == True:
            count += 1
        print("预测分数为：",int(predict[i]),'\t',"实际分数为：", dev_label[i])
    print("准确率为:",count / len(predict))

def testError(predict, true_label):
    if abs(predict - true_label) <= 5 :
        return True

def getscores(vec, lr_model,feature, articles):
    content = articles.replace(" ", "").replace("　", "").replace("", "").replace("&nbsp;", "").replace("<br>", "")
    tokens = [word for word in jieba.cut(content)]
    dev_data = [" ".join(tokens)]
    dev = vec.transform(dev_data)
    dev_word_sizes, dev_sen_sizes, dev_highword_sizes, dev_wc_sizes, dev_im_sizes, dev_conj_sizes, dev_yinhao_sizes, dev_shuminghao_sizes= getFeature(dev_data,
                                                                                                            feature)
    dev = csr_matrix(dev).toarray()
    dev = np.hstack(
        (dev, dev_word_sizes, dev_sen_sizes, dev_highword_sizes, dev_wc_sizes, dev_im_sizes, dev_conj_sizes, dev_yinhao_sizes, dev_shuminghao_sizes))
    predict = lr_model.predict(dev)
    error = random.randint(-5, 5)
    result = predict[0] + error
    if result >40:
        result = 40
    return result

if __name__ == '__main__':
    getTfidf()
    # train_svm()
    # testmatrix()
    pass
