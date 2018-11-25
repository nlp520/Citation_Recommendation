#!usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on 2018年5月1日
@author: sui
'''
import gensim
from gensim.models import word2vec
import logging
import pickle
from process import pickleload
from tqdm import tqdm
import re

def data_process(sentence):
    fl = re.findall('[a-z]\.[A-Z]', sentence)
    for f in fl:
        new_f = ""
        for w in f:
            new_f += w + " "
        new_f = new_f.strip(" ")
        sentence = sentence.replace(f, new_f)
    sentence = sentence.replace(' ’',"’")
    return sentence

def writeFile(file, content):
    with open(file, 'a') as fp:
        fp.write(content)

def getWord2vecData():
    '''
        [
                {
                 "citStr":"" 引用的作者和年份,
                 "context":"", 整个引用片段
                 "up_source_tokens":"",
                 "down_source_tokens":"",
                 "target_tokens":""
                 "citations":[
                                {
                                "up_source_tokens":"",
                                "down_source_tokens":"",
                                "target_tokens":""
                                }
                               ...
                              ]
                }
                ......

            ]
        查找相似citation
        :return:
        '''
    datas = pickleload("../data2/train_data.pkl", "./data2/train_data.pkl")
    # datas = datas[len(datas)-1000:len(datas)]
    print(len(datas))
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])
        up_content = data_process(data['up_source_tokens'])
        down_content = data_process(data['down_source_tokens'])
        writeFile('./word2vec/train_word2vec.txt', up_content + " " + target + " "+ down_content + "\n")

def train():
    #打印日志信息
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    #添加数据集
    sentences = word2vec.Text8Corpus("./word2vec/train_word2vec.txt")   # word_seg词级别
    #构建模型
    wordvecmodel = word2vec.Word2Vec(sentences,size=300,min_count=1,iter=10)
    #模型保存
    wordvecmodel.save("./word2vec/word2vec_300.model")#单词385787
    
def similar():
#     model = gensim.models.Word2Vec.load("./model/word2vec_100.model")
#     word2vec_path = "./word2vec/gloveModel300d.txt"
    # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    model = gensim.models.Word2Vec.load("./word2vec/word2vec_300.model")
    words = model.most_similar("美朝")
    for word in words:
        print (word[0],word[1])

def generateword2vecpkl():
    '''
    wv = KeyedVectors()
    wv.syn0 = self.__dict__.get('syn0', [])
    wv.syn0norm = self.__dict__.get('syn0norm', None)
    wv.vocab = self.__dict__.get('vocab', {})
    wv.index2word = self.__dict__.get('index2word', [])
    :return:
    '''
    model = gensim.models.Word2Vec.load("./word2vec/word2vec_300.model")
    dic = {}
    index2word = {}
    for i in range(len(model.wv.index2word)):
        dic[model.wv.index2word[i]] = i + 2
        index2word[i+2] = model.wv.index2word[i]
    index2word[0] = "padding"
    index2word[1] = "<unknow>"
    pickle.dump(dic, open('./word2vec/word2vec_word2index_300.pkl', 'wb'))
    pickle.dump(index2word, open('./word2vec/word2vec_index2word_300.pkl', 'wb'))
    print(len(dic))

def generateEmbedding():
    import numpy as np
    embedding_vec = []
    model = gensim.models.Word2Vec.load("./word2vec/word2vec_300.model")
    dic = pickle.load(open('./word2vec/word2vec_word2index_300.pkl', 'rb'))
    index2word_dic = pickle.load(open('./word2vec/word2vec_index2word_300.pkl', 'rb'))
    unknow = np.zeros(300)
    padding = np.zeros(300)
    embedding_vec.append(padding)
    embedding_vec.append(unknow)
    for i in range(2, len(index2word_dic)):
        embedding_vec.append(model.wv[index2word_dic[i]])

    dic['padding'] = 0
    dic['<unknow>'] = 1
    pickle.dump(embedding_vec, open('./word2vec/embedding_vec_300.pkl', 'wb'))

if __name__ == '__main__':#177699
    # getWord2vecData()
    # train()
    generateword2vecpkl()
    generateEmbedding()
    pass



