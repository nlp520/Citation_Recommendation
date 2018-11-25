#-*- coding:utf-8 -*-
from process import pickleload, jsonsave, picklesave
from rougetest.test_bleu import test_bleu
from collections import OrderedDict
from copy import copy
from rougetest.test_score import test_score
from tqdm import tqdm
import pandas as pd
import re

from similar import getSVMScore

def process_kuohao(sentence):
    '''
    对输入的句子进行处理，去掉里面的（）
    :param sentence:
    :return:
    '''
    fl = re.findall('(?<=\\()[^\\)]+', sentence)
    for f in fl:
        f = "(" + f + ")"
        sentence = sentence.replace(f, " ")
    fl = re.findall('\[(.*?)\]', sentence)
    for f in fl:
        f = "[" + f + "]"
        sentence = sentence.replace(f, " ")
    sentence = sentence.replace("   "," ").replace("  "," ")
    # print(sentence)
    return sentence

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

def getSingleTrainData():
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
    new_datas = copy(datas)
    train_datas = []
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])

        #计算citation
        citations = data["citations_tokens"]
        scores = []
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = data_process(ciation["target_tokens"])
            score = test_bleu(cit_target, target, 1)
            scores.append(score)
            new_datas[i]['citations_tokens'][index]["bleu1_score"] = score

            dic = {}
            dic['up_source'] = data_process(data["up_source_tokens"])
            dic['down_source'] = data_process(data["down_source_tokens"])
            dic['target'] = data_process(data["target_tokens"])
            dic['cit_up_source'] = data_process(ciation['up_source_tokens'])
            dic['cit_down_source'] = data_process(ciation['down_source_tokens'])
            dic['cit_target'] = data_process(ciation['target_tokens'])
            dic['bleu1_score'] = score
            if score == 1:
                continue
            train_datas.append(copy(dic))
    print("训练样本的数据量为：", len(train_datas))
    picklesave(train_datas, "./train_data/single_train_data.pkl", "single_train_data.pkl")

def getBigPairsTrainData():
    import random
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
    datas = pickleload("../data2/train_data2.pkl", "./data2/train_data2.pkl")
    idf_dic = pickleload("../data2/idf.pkl", "idf.pkl")
    # datas = datas[len(datas)-1000:len(datas)]
    print(len(datas))
    train_datas = []
    train_datas2 = []
    train_spill = []
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        scores = []
        if len(target) < 50:
            continue
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = data_process(ciation["target_tokens"])
            if target == cit_target or len(cit_target) < 50:
                scores.append(0)
            else:
                score = getSVMScore(idf_dic, process_kuohao(target), process_kuohao(cit_target))
                scores.append(score)

        sorted_scores = sorted(scores, reverse=True)
        best_indexs = []
        for j in range(len(sorted_scores)):
            if sorted_scores[j]>0.1 and j <=5:
                best_index = scores.index(sorted_scores[j])
                best_indexs.append(best_index)
        if len(best_indexs) == len(citations):
            continue
        for best_index in best_indexs:
            train_data = {}
            train_data['up_source'] = data_process(data["up_source_tokens"])
            train_data['down_source'] = data_process(data["down_source_tokens"])
            train_data['target'] = data_process(data["target_tokens"])

            high_dic = {}
            high_dic['cit_up_source'] = data_process(citations[best_index]['up_source_tokens'])
            high_dic['cit_down_source'] = data_process(citations[best_index]['down_source_tokens'])
            high_dic['cit_target'] = data_process(citations[best_index]['target_tokens'])
            high_dic['bleu1_score'] = scores[best_index]

            # for k in range(len(best_indexs)):
            #     print("target:", train_data['target'])
            #     print("cit_target:", data_process(citations[best_indexs[k]]['target_tokens']))
            #     print("score:", sorted_scores[k])
            #     print("\n")
            # print(len(best_indexs), "  /   ", len(citations))
            # print("---------------------------------------------")
            for low_index in range(len(scores)):
                if low_index not in best_indexs:
                    if scores[best_index] == scores[low_index] or scores[best_index] == 1.0:
                        continue
                    low_dic = {}
                    low_dic['cit_up_source'] = data_process(citations[low_index]['up_source_tokens'])
                    low_dic['cit_down_source'] = data_process(citations[low_index]['down_source_tokens'])
                    low_dic['cit_target'] = data_process(citations[low_index]['target_tokens'])
                    low_dic['bleu1_score'] = scores[low_index]
                    if low_dic['cit_target'] == train_data['target']:
                        continue
                    train_data['high_dic'] = high_dic
                    train_data['low_dic'] = low_dic
                    train_spill.append(train_data)

        if i in [len(datas) // 5, len(datas) * 2 // 5, len(datas) * 3 // 5, len(datas) * 4 // 5, len(datas) - 1]:
            train_datas.append(train_spill)
            print(len(train_spill))
            train_spill = []

    print(len(train_datas))
    print(len(train_datas2)) #26933
    print("训练样本的数据量为：", len(train_datas))
    picklesave(train_datas, "./train_data/big_pairs_train_data.pkl", "big_pairs_train_data.pkl")
    # picklesave(train_datas2, "../data2/train_data2.pkl", "./data2/train_data2.pkl")

def filterData():
    import random
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
    idf_dic = pickleload("../data2/idf.pkl", "idf.pkl")
    # datas = datas[len(datas)-1000:len(datas)]
    print(len(datas))
    train_datas = []
    train_datas2 = []
    train_spill = []
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        scores = []
        if len(target) < 50:
            continue
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = data_process(ciation["target_tokens"])
            if target == cit_target or len(cit_target) < 50:
                scores.append(0)
            else:
                score = getSVMScore(idf_dic, process_kuohao(target), process_kuohao(cit_target))
                scores.append(score)

        sorted_scores = sorted(scores, reverse=True)
        best_indexs = []
        score_filter = 0.1
        for j in range(len(sorted_scores)):
            if sorted_scores[j] > 0.1 and j<=5 :
                best_index = scores.index(sorted_scores[j])
                best_indexs.append(best_index)
                score_filter = sorted_scores[j]

        train_data = {}
        train_data['up_source'] = data_process(data["up_source_tokens"])
        train_data['down_source'] = data_process(data["down_source_tokens"])
        train_data['target'] = data_process(data["target_tokens"])

        best_index = scores.index(sorted_scores[0])
        high_dic = {}
        high_dic['cit_up_source'] = data_process(citations[best_index]['up_source_tokens'])
        high_dic['cit_down_source'] = data_process(citations[best_index]['down_source_tokens'])
        high_dic['cit_target'] = data_process(citations[best_index]['target_tokens'])
        high_dic['bleu1_score'] = scores[best_index]

        if sorted_scores[0] < 0.1:
            continue
        # for k in range(len(best_indexs)):
        #     print("target:", train_data['target'])
        #     print("cit_target:", data_process(citations[best_indexs[k]]['target_tokens']))
        #     print("score:", sorted_scores[k])
        #     print("\n")
        # print(len(best_indexs), "  /   ", len(citations))
        # print("---------------------------------------------")
        low_index = random.randint(0, len(scores)-1)
        while low_index == best_index :
            low_index = random.randint(0, len(scores) - 1)
        if scores[best_index] == scores[low_index] or scores[best_index] == 1.0:
            continue
        low_dic = {}
        low_dic['cit_up_source'] = data_process(citations[low_index]['up_source_tokens'])
        low_dic['cit_down_source'] = data_process(citations[low_index]['down_source_tokens'])
        low_dic['cit_target'] = data_process(citations[low_index]['target_tokens'])
        low_dic['bleu1_score'] = scores[low_index]
        if low_dic['cit_target'] == train_data['target']:
            continue

        new_data = {}
        new_data['target_tokens'] = data_process(data["target_tokens"])
        new_data['up_source_tokens'] = data_process(data["up_source_tokens"])
        new_data['down_source_tokens'] = data_process(data["down_source_tokens"])
        new_citations = []
        for citation in citations:
            cit_dic = {}
            cit_dic['up_source_tokens'] = data_process(citation['up_source_tokens'])
            cit_dic['down_source_tokens'] = data_process(citation['down_source_tokens'])
            cit_dic['target_tokens'] = data_process(citation['target_tokens'])
            if getSVMScore(idf_dic, process_kuohao(target), process_kuohao(cit_dic['target_tokens'])) >=score_filter:
                cit_dic['label'] = 1
            else:
                cit_dic['label'] = 0
            if cit_dic['target_tokens'] == new_data['target_tokens']:
                continue
            new_citations.append(cit_dic)

        if len(new_citations) < 5:
            continue
        new_data['citations_tokens'] = new_citations
        train_datas2.append(new_data)
        train_data['high_dic'] = high_dic
        train_data['low_dic'] = low_dic
        train_spill.append(train_data)
        if i in [len(datas)//5,len(datas) *2//5, len(datas) *3//5,len(datas) *4//5 ,len(datas)-1]:
            train_datas.append(train_spill)
            train_spill = []

    print(len(train_datas))
    print(len(train_datas2)) #26933
    print("训练样本的数据量为：", len(train_datas))
    # picklesave(train_datas, "./train_data/small_pairs_train_data2.pkl", "small_pairs_train_data2.pkl")
    picklesave(train_datas2, "../data2/train_data2.pkl", "./data2/train_data2.pkl")

def getSmallPairsTrainData():
    import random
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
    datas = pickleload("../data2/train_data2.pkl", "./data2/train_data2.pkl")
    idf_dic = pickleload("../data2/idf.pkl", "idf.pkl")
    # datas = datas[len(datas)-1000:len(datas)]
    print(len(datas))
    train_datas = []
    train_datas2 = []
    train_spill = []
    q_id = 0
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        scores = []
        if len(target) < 50:
            continue
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = data_process(ciation["target_tokens"])
            if target == cit_target or len(cit_target) < 50:
                scores.append(0)
            else:
                score = getSVMScore(idf_dic, process_kuohao(target), process_kuohao(cit_target))
                scores.append(score)

        sorted_scores = sorted(scores, reverse=True)
        best_indexs = []
        for j in range(len(sorted_scores)):
            if sorted_scores[j]>0.1 and j <=5:
                best_index = scores.index(sorted_scores[j])
                best_indexs.append(best_index)
        if len(best_indexs) == len(citations):
            continue
        for best_index in best_indexs:
            train_data = {}
            train_data['up_source'] = data_process(data["up_source_tokens"])
            train_data['down_source'] = data_process(data["down_source_tokens"])
            train_data['target'] = data_process(data["target_tokens"])

            high_dic = {}
            high_dic['cit_up_source'] = data_process(citations[best_index]['up_source_tokens'])
            high_dic['cit_down_source'] = data_process(citations[best_index]['down_source_tokens'])
            high_dic['cit_target'] = data_process(citations[best_index]['target_tokens'])
            high_dic['bleu1_score'] = scores[best_index]

            # for k in range(len(best_indexs)):
            #     print("target:", train_data['target'])
            #     print("cit_target:", data_process(citations[best_indexs[k]]['target_tokens']))
            #     print("score:", sorted_scores[k])
            #     print("\n")
            # print(len(best_indexs), "  /   ", len(citations))
            # print("---------------------------------------------")
            low_index = random.randint(0, len(scores) - 1)
            while low_index in best_indexs:
                low_index = random.randint(0, len(scores) - 1)
            if scores[best_index] == scores[low_index] or scores[best_index] == 1.0:
                continue
            low_dic = {}
            low_dic['cit_up_source'] = data_process(citations[low_index]['up_source_tokens'])
            low_dic['cit_down_source'] = data_process(citations[low_index]['down_source_tokens'])
            low_dic['cit_target'] = data_process(citations[low_index]['target_tokens'])
            low_dic['bleu1_score'] = scores[low_index]
            if low_dic['cit_target'] == train_data['target']:
                continue
            train_data['high_dic'] = high_dic
            train_data['low_dic'] = low_dic
            train_spill.append(train_data)

        if i in [len(datas) // 5, len(datas) * 2 // 5, len(datas) * 3 // 5, len(datas) * 4 // 5, len(datas) - 1]:
            train_datas.append(train_spill)
            print(len(train_spill))
            train_spill = []

    print(len(train_datas))
    print(len(train_datas2)) #26933
    print("训练样本的数据量为：", len(train_datas))
    picklesave(train_datas, "./train_data/small_pairs_train_data.pkl", "small_pairs_train_data.pkl")
    # picklesave(train_datas2, "../data2/train_data2.pkl", "./data2/train_data2.pkl")

def getZooEmbedding():
    source_embedding = pickleload("./word2vec/glove_300.pkl", "glove_300.pkl")
    for i in range(len(source_embedding)):
        print(i)
        str_embedding = [str(j) for j in source_embedding[i]]
        writefile("./match_zoo_data/embedding_dic.txt", str(i)+" "+ " ".join(str_embedding) + "\n")


def getMatchZooData():
    import random
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
    datas = pickleload("../data2/random_train_data.pkl", "./data2/random_train_data.pkl")
    word2index = pickleload("./word2vec/glove_word2index_300.pkl", "./word2vec/glove_word2index_300.pkl")
    print(len(datas))
    q_id = 0
    for i in tqdm(range(len(datas))):
        data = datas[i]
        source_tokens = data_process(data["up_source_tokens"]) +" "+ data_process(data["up_source_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        writefile('./match_zoo_data/corpus_preprocessed.txt', "Q_"+str(q_id) + "\t250\t" + getSen_index(source_tokens, word2index) + "\n")

        d_id = 0
        for citation in citations:
            score = citation['label']
            citation_tokens = data_process(citation["up_source_tokens"]) +" "+data_process(citation["target_tokens"]) +" "+ data_process(citation["up_source_tokens"])
            writefile('./match_zoo_data/corpus_preprocessed.txt',
                      "Q_" + str(q_id) +"D_" + str(d_id) + "\t250\t" + getSen_index(citation_tokens, word2index) + "\n")
            if q_id < len(datas) * 4 // 5:
                writefile('./match_zoo_data/relation_train.txt',
                          str(score) + "\t" + "Q_" + str(q_id)+"\t" + "Q_" + str(q_id) + "D_" + str(d_id) + "\n" )
            else:
                writefile('./match_zoo_data/relation_test.txt',
                          str(score) + "\t" + "Q_" + str(q_id) + "\t" + "Q_" + str(q_id) + "D_" + str(d_id) + "\n")
                writefile('./match_zoo_data/relation_valid.txt',
                          str(score) + "\t" + "Q_" + str(q_id) + "\t" + "Q_" + str(q_id) + "D_" + str(d_id) + "\n")
            d_id += 1
        q_id += 1

def getindex(word, word2index):
    if word in word2index:
        return str(word2index[word])
    else:
        return "1"

def getSen_index(source_tokens, word2index):
    source_tokens = source_tokens.split(" ")
    new_source_tokens = [getindex(word, word2index) for word in source_tokens]
    return " ".join(new_source_tokens)

def writefile(path, content):
    with open(path, "a") as fp:
        fp.write(content)

def countScores():
    datas = pickleload("./train_data/train_data.pkl", "./train_data/train_data.pkl")
    for data in datas:
        score = data['bleu1_score']
        print(score)

def getCsvFile():
    datas = pickleload("./train_data/train_data.pkl", "./train_data/train_data.pkl")
    content_source = []
    target = []
    cit_content_source = []
    cit_target = []
    score = []
    for data in tqdm(datas):
        content_source.append(data['up_source'] + ' ' + data['down_source'])
        target.append(data['target'])
        cit_content_source.append(data['cit_up_source'] + ' ' + data['cit_down_source'])
        cit_target.append(data['cit_target'])
        score.append(data['bleu1_score'])
    train_data = pd.concat([pd.DataFrame(data = content_source, columns=['content_source']),\
                            pd.DataFrame(data = target, columns=['target']), \
                            pd.DataFrame(data=cit_content_source,columns=['cit_content_source']), pd.DataFrame(data = cit_target, columns=['cit_target']),\
                            pd.DataFrame(data = score, columns=['score'])], axis = 1)
    train_data.to_csv("./train_data/train_data.csv")

def getsomeSample():
    datas = pickleload("../data2/train_data.pkl", "./data2/train_data.pkl")
    for i in tqdm(range(len(datas))):
        data = datas[i]
        target = data_process(data["target_tokens"])
        up_source_tokens = data_process(data["up_source_tokens"])
        down_source_tokens = data_process(data["down_source_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        print("up_context:", up_source_tokens)
        print("down_context:", down_source_tokens)
        print("target_citation:", target)
        scores = []
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = data_process(ciation["target_tokens"])
            print(index, " citations:", cit_target)
        print("--------------------------------------------")

def manual_label():
    datas = pickleload("../data2/train_data2.pkl", "./data2/train_data.pkl")
    # golden_train_datas = pickleload("../data/golden_train_data.pkl", "./data/golden_train_data.pkl")
    print(len(datas))
    train_datas = []
    flag_pairs = {}
    for i in range(len(datas)):
        data = datas[i]
        target = data_process(data["target_tokens"])
        # 计算citation
        citations = data["citations_tokens"]
        flag = 0
        for index in range(len(citations)):
            citation = citations[index]
            cand_cit = data_process(citation["target_tokens"])
            if cand_cit + target not in flag_pairs.keys():
                print("进程：" , i ,"/" ,len(datas), "  ",  index, "/",  len(citations))
                print("target:", target)
                print("candidate:", cand_cit)
                label = input("标签：")
                if str(label) == "1":
                    citations[index]['label'] = 1
                    flag = 1
                else:
                    citations[index]['label'] = 0
                flag_pairs[cand_cit + target] = citations[index]['label']
                flag_pairs[target + cand_cit] = citations[index]['label']
            else:
                if flag_pairs[cand_cit + target] == 1:
                    citations[index]['label'] = 1
                    flag = 1
                else:
                    citations[index]['label'] = 0
        picklesave(flag_pairs, "../data/flag_pairs.pkl", "./data/flag_pairs.pkl")
        if flag == 1:
            new_data = datas[i]
            new_data["citations_tokens"] = citations
            train_datas.append(new_data)
            picklesave(train_datas, "../data/golden_train_data.pkl", "./data/golden_train_data.pkl")
            # x = pickleload("../data/golden_train_data.pkl", "./data/golden_train_data.pkl")
            # print(x)

def statistical_data():
    datas = pickleload("../data2/train_data2.pkl", "./data2/train_data.pkl")
    label_num_dic = {}
    for data in datas:
        citations = data['citations_tokens']
        count = 0
        for citation in citations:
            if citation["label"] == 1:
                count += 1
        if count not in label_num_dic:
            label_num_dic[count] = 1
        else:
            label_num_dic[count] +=1
    for key, value in label_num_dic.items():
        print(key, " : ",value )

def getRandomData():
    import numpy as np
    datas = pickleload("../data2/train_data2.pkl", "./data2/train_data2.pkl")
    new_datas = []
    ids = range(len(datas))
    permutation = np.random.permutation(ids)
    for i, id in enumerate(permutation):
        new_datas.append(datas[id])
    picklesave(new_datas ,"../data2/random_train_data.pkl", "./data2/random_train_data.pkl")

if __name__ == '__main__':
    getsomeSample()
    # getMatchZooData()
    # getZooEmbedding()
    pass

