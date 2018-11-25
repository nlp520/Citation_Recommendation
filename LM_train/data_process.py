#-*- coding:utf-8 -*-
from process import pickleload, jsonsave, picklesave
from tqdm import tqdm
import pandas as pd
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

def getWord2index():
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
    tokenDic ={}
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"]).split(" ")
        up_source = data_process(data["up_source_tokens"]).split(" ")
        down_source = data_process(data["down_source_tokens"]).split(" ")
        word_lis = target + up_source + down_source
        for token in word_lis:
            if token not in tokenDic:
                tokenDic[token] = 1
            else:
                tokenDic[token] += 1

    index = 2
    word2index = {}
    for key, value in tokenDic.items():
        if value > 1:
            word2index[key] = index
            index += 1
    word2index['<padding>'] = 0
    word2index['<unknow>'] = 1
    word2index['<CLS>'] = index
    word2index['<DSP>'] = index + 1
    word2index['<MASK>'] = index + 2
    print(len(word2index),"  /  ", len(tokenDic), "个token")
    picklesave(word2index, './word_vec/word2index.pkl', "word2index.pkl")

if __name__ == '__main__':
    getWord2index()
    pass

