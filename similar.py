#!usr/bin/python
from collections import OrderedDict
import pickle
from tqdm import tqdm
from Retrieval.train import cal_MAP
from process import pickleload, jsonsave, picklesave
import math
from rougetest.test_bleu import getBleu, test_bleu
from rougetest.test_score import test_score
import re
import nltk
def process(sentence):
    fl = re.findall('[a-z]\.[A-Z]', sentence)
    for f in fl:
        new_f = ""
        for w in f:
            new_f += w + " "
        new_f = new_f.strip(" ")
        sentence = sentence.replace(f, new_f)
    sentence = sentence.replace(' ’',"’")
    return sentence

def getIdf():
    datas = pickleload("./data2/train_data.pkl", "./data/train_data.pkl")
    all_count = len(datas)
    print(len(datas))
    tokenidf_dic = {}
    for data in tqdm(datas):
        up_source_tokens = process(data["up_source_tokens"]).split(" ")
        down_source_tokens = process(data["down_source_tokens"]).split(" ")
        target_tokens = process(data["target_tokens"]).split(" ")
        dic = {}
        for token in up_source_tokens + down_source_tokens + target_tokens:
            if token not in dic:
                dic[token] = 1

        for key in dic.keys():
            if key not in tokenidf_dic:
                tokenidf_dic[key] = 1
            else:
                tokenidf_dic[key] += 1
    new_dic = {}
    for key, value in tokenidf_dic.items():
        new_dic[key] = math.log10(all_count/value)
    picklesave(new_dic, './data2/idf.pkl', "idf")

def getIdf2():
    citationDic = pickle.load(open("./data/processed_data3.pkl", "rb"))
    all_count = 0
    tokenidf_dic = {}
    for key, value in tqdm(citationDic.items()):
        all_count += len(value)
        for v in value:
            context = v['context']
            word_lis = nltk.word_tokenize(context)
            dic = {}
            for token in word_lis:
                if token not in dic:
                    dic[token] = 1

            for key in dic.keys():
                if key not in tokenidf_dic:
                    tokenidf_dic[key] = 1
                else:
                    tokenidf_dic[key] += 1
    new_dic = {}
    for key, value in tokenidf_dic.items():
        new_dic[key] = math.log10(all_count/value)
    picklesave(new_dic, './data2/idf2.pkl', "idf")

def findSimilar():
    '''
    [
            {
             "citStr":"" 引用的作者和年份,
             "context":"", 整个引用片段
             "up_source":"",
             "down_source":"",
             "target":""
             "citations":[
                          citation0,
                          citation1,
                           ...
                          ]
            }
            ......

        ]
    查找相似citation
    :return:
    '''
    datas = pickleload("./data2/random_train_data.pkl", "./data2/random_train_data.pkl")
    datas = datas[len(datas)*4//5:len(datas)]
    idf_dic = pickleload("./data2/idf.pkl", "idf.pkl")
    # datas = datas[0:10]
    print(len(idf_dic))
    print(len(datas))
    count = 0

    MAPS = 0
    precisions = 0
    recalls = 0
    for data in tqdm(datas):
        up_source_tokens = process(data["up_source_tokens"])
        down_source_tokens = process(data["down_source_tokens"])
        target = process(data["target_tokens"])

        #计算citation
        citations = data["citations_tokens"]
        scores = []
        count += 1
        ref_lis = []
        for index in range(len(citations)):
            if citations[index]['label'] == 1:
                ref_lis.append(index)
            ciation = citations[index]
            cit_up_source_tokens = process(ciation["up_source_tokens"])
            cit_down_source_tokens = process(ciation["down_source_tokens"])
            cit_target = process(ciation["target_tokens"])
            score = getSVMScore(idf_dic, up_source_tokens, cit_up_source_tokens + " " + cit_target +" "+cit_down_source_tokens)
            scores.append(score)
        # print("scores:",scores)
        new_score = sorted(scores,reverse = True)
        pre_lis = []
        for i in range(3):
            pre_lis.append(scores.index(new_score[i]))
        # print("原文：",up_source_tokens + " "+ down_source_tokens)
        # print("候选：",citations[pre_lis[0]]["up_source_tokens"])
        # print("候选：",citations[pre_lis[0]]["target_tokens"])
        # print("候选：",citations[pre_lis[0]]["down_source_tokens"])
        # print("ref_lis",ref_lis)
        # print("pre_lis",pre_lis)
        precision, recall, MAP = cal_MAP(ref_lis, pre_lis)
        # print("precision:", precision)
        # print("recall:", recall)
        # print("MAP:", MAP)
        # print("-----------------------------------------------")
        MAPS += MAP
        precisions += precision
        recalls += recall

    MAPS /= len(datas)
    precisions /= len(datas)
    recalls /= len(datas)
    print("MAP：%.4f  P：%.4f  R：%.4f"%(MAPS, precisions, recalls))

def findVSMSimilar():
    '''
    [
            {
             "citStr":"" 引用的作者和年份,
             "context":"", 整个引用片段
             "up_source":"",
             "down_source":"",
             "target":""
             "citations":[
                          citation0,
                          citation1,
                           ...
                          ]
            }
            ......

        ]
    查找相似citation
    :return:
    '''
    datas = pickleload("./data2/random_train_data.pkl", "./data/train_data.pkl")
    datas = datas[len(datas)*4//5:len(datas)]
    idf_dic = pickleload("./data2/idf.pkl", "idf.pkl")
    # datas = datas[0:10]
    print(len(idf_dic))
    print(len(datas))
    result_lis = []
    count = 0
    for data in datas:
        up_source_tokens = process(data["up_source_tokens"]).split(" ")
        down_source_tokens = process(data["down_source_tokens"]).split(" ")
        target = process(data["target_tokens"])
        dic = {}
        for token in up_source_tokens:
            if token not in dic:
                dic[token] = 1
            else:
                dic[token] += 1
        for token in down_source_tokens:
            if token not in dic:
                dic[token] = 1
            else:
                dic[token] += 1
        keys = dic.keys()
        sqrt_source = 0.0
        for key in keys:
            if key in idf_dic:
                dic[key] = dic[key]/(len(up_source_tokens) + len(down_source_tokens)) * idf_dic[key]
            else:
                dic[key] = 0
            sqrt_source += dic[key] * dic[key]
        sqrt_source = math.sqrt(sqrt_source)

        #计算citation
        citations = data["citations_tokens"]
        scores = []
        # if len(citations) < 20:
        #     continue
        count += 1
        for index in range(len(citations)):
            ciation = citations[index]
            cit_up_source_tokens = ciation["up_source_tokens"].split(" ")
            cit_down_source_tokens = ciation["down_source_tokens"].split(" ")
            cit_target = process(ciation["target_tokens"]).split(" ")
            cit_dic = {}
            for token in cit_target:
                if token not in cit_dic:
                    cit_dic[token] = 1
                else:
                    cit_dic[token] += 1
            keys = cit_dic.keys()
            sqrt_cit = 0.0
            for key in keys:
                if key in idf_dic:
                    cit_dic[key] = cit_dic[key] / (len(cit_target)) * idf_dic[key]
                else:
                    cit_dic[key] = 0
                sqrt_cit += cit_dic[key]
            sqrt_cit = math.sqrt(sqrt_cit)
            #计算相似度
            sum = 0.0
            for key in dic.keys():
                if key in cit_dic:
                    sum += dic[key] * cit_dic[key]

            score = sum/(sqrt_source * sqrt_cit)
            scores.append(score)
        new_score = sorted(scores,reverse = True)

        best_index = scores.index(new_score[0])
        predict = citations[best_index]['target_tokens']
        result_dic = OrderedDict()
        result_dic["cand_answer"] = predict
        result_dic["ref_answer"] = target
        result_lis.append(result_dic)
        # print("上文:",data["up_source_tokens"])
        # print("下文:",data["down_source_tokens"])
        # print("原始的：", target)
        # print("预测的：", predict)
        # print("-------------------------------------------------------")
    print(count ,"   /    ", len(datas), "   /    ",count/len(datas))
    jsonsave('./rougetest/data/similar_data.json', result_lis, "result_lis")
    test_score("./rougetest/data/similar_data.json", n_size=1)
    test_score("./rougetest/data/similar_data.json", n_size=2)
    test_score("./rougetest/data/similar_data.json", n_size=3)
    test_score("./rougetest/data/similar_data.json", n_size=4)

def findBleuSimilar():
    '''
    [
            {
             "citStr":"" 引用的作者和年份,
             "context":"", 整个引用片段
             "up_source":"",
             "down_source":"",
             "target":""
             "citations":[
                          citation0,
                          citation1,
                           ...
                          ]
            }
            ......

        ]
    查找相似citation
    :return:
    '''
    datas = pickleload("./data2/train_data.pkl", "./data2/train_data.pkl")
    # datas = datas[len(datas)-1000:len(datas)]
    print(len(datas))
    result_lis = []
    count = 0
    for data in datas:
        target = data["target_tokens"].split(" ")

        #计算citation
        citations = data["citations_tokens"]
        scores = []
        for index in range(len(citations)):
            ciation = citations[index]
            cit_target = ciation["target_tokens"].split(" ")
            score = test_bleu(" ".join(cit_target), " ".join(target), 1)
            scores.append(score)
        new_score = sorted(scores,reverse = True)

        # if new_score[0] < 0.5 and new_score[0] != 1:
        #     continue
        best_index = scores.index(new_score[0])
        predict = citations[best_index]['target_tokens']
        result_dic = OrderedDict()
        result_dic["cand_answer"] = predict
        result_dic["ref_answer"] = data["target_tokens"]
        result_lis.append(result_dic)
        count += 1
        print("score:", new_score[0])
        print("原始的：", data["target_tokens"])
        print("预测的：", predict)
        print("-------------------------------------------------------")
    print(count)
    jsonsave('./rougetest/data/target_data.json', result_lis, "result_lis")
    test_score("./rougetest/data/target_data.json", n_size=1)
    test_score("./rougetest/data/target_data.json", n_size=2)
    test_score("./rougetest/data/target_data.json", n_size=3)
    test_score("./rougetest/data/target_data.json", n_size=4)

def getTopSimilar(data, idf_dic, top=1):
    up_source_tokens = data["up_source_tokens"].split(" ")
    down_source_tokens = data["down_source_tokens"].split(" ")
    target = data["target_tokens"]
    dic = {}
    for token in up_source_tokens:
        if token not in dic:
            dic[token] = 1
        else:
            dic[token] += 1
    for token in down_source_tokens:
        if token not in dic:
            dic[token] = 1
        else:
            dic[token] += 1
    keys = dic.keys()
    sqrt_source = 0.0
    for key in keys:
        if key in idf_dic:
            dic[key] = dic[key] / (len(up_source_tokens) + len(down_source_tokens)) * idf_dic[key]
        else:
            dic[key] = 0
        sqrt_source += dic[key] * dic[key]
    sqrt_source = math.sqrt(sqrt_source)

    # 计算citation
    citations = data["citations_tokens"]
    scores = []
    top = len(citations) if len(citations) < top else top
    for index in range(len(citations)):
        ciation = citations[index]
        cit_up_source_tokens = ciation["up_source_tokens"].split(" ")
        cit_down_source_tokens = ciation["down_source_tokens"].split(" ")
        cit_target = ciation["target_tokens"]
        cit_dic = {}
        for token in cit_up_source_tokens:
            if token not in cit_dic:
                cit_dic[token] = 1
            else:
                cit_dic[token] += 1
        for token in cit_down_source_tokens:
            if token not in cit_dic:
                cit_dic[token] = 1
            else:
                cit_dic[token] += 1
        keys = cit_dic.keys()
        sqrt_cit = 0.0
        for key in keys:
            if key in idf_dic:
                cit_dic[key] = cit_dic[key] / (len(up_source_tokens) + len(down_source_tokens)) * idf_dic[key]
            else:
                cit_dic[key] = 0
            sqrt_cit += cit_dic[key]
        sqrt_cit = math.sqrt(sqrt_cit)
        # 计算相似度
        sum = 0.0
        for key in dic.keys():
            if key in cit_dic:
                sum += dic[key] * cit_dic[key]

        score = sum / (sqrt_source * sqrt_cit)
        scores.append(score)
    new_score = sorted(scores, reverse=True)
    result_lis = []
    for i in range(top):
        best_index = scores.index(new_score[i])
        predict = citations[best_index]
        result_lis.append(predict)
    return result_lis

def getSVMScore(idf_dic, target, candidate):
    dic = {}
    target = [token for token in target.split(" ") if token not in [' ', '']]
    candidate = [token for token in candidate.split(" ") if token not in [' ', '', ".", ","]]
    for token in target:
        if token not in dic:
            dic[token] = 1
        else:
            dic[token] += 1
    keys = dic.keys()
    sqrt_source = 0.0
    for key in keys:
        if key in idf_dic:
            dic[key] = dic[key] / (len(target)) * idf_dic[key]
        else:
            dic[key] = 0
        sqrt_source += dic[key] * dic[key]
    sqrt_source = math.sqrt(sqrt_source)
    cit_dic = {}
    for token in candidate:
        if token not in cit_dic:
            cit_dic[token] = 1
        else:
            cit_dic[token] += 1
    keys = cit_dic.keys()
    sqrt_cit = 0.0
    for key in keys:
        if key in idf_dic:
            cit_dic[key] = cit_dic[key] / (len(candidate)) * idf_dic[key]
        else:
            cit_dic[key] = 0
        sqrt_cit += cit_dic[key]
    sqrt_cit = math.sqrt(sqrt_cit)
    # 计算相似度
    sum = 0.0
    for key in dic.keys():
        if key in cit_dic:
            sum += dic[key] * cit_dic[key]
    if sqrt_source * sqrt_cit == 0:
        score = 0
    else:
        score = sum / (sqrt_source * sqrt_cit)
    return score

def getTopVsmScore(idf_dic, target, candidates, num=3):
    dic = {}
    target = target.split(" ")

    for token in target:
        if token not in dic:
            dic[token] = 1
        else:
            dic[token] += 1
    keys = dic.keys()
    sqrt_source = 0.0
    for key in keys:
        if key in idf_dic:
            dic[key] = dic[key] / (len(target)) * idf_dic[key]
        else:
            dic[key] = 0
        sqrt_source += dic[key] * dic[key]
    sqrt_source = math.sqrt(sqrt_source)
    scores = []
    for candidate in candidates:
        candidate = candidate.split(" ")
        cit_dic = {}
        for token in candidate:
            if token not in cit_dic:
                cit_dic[token] = 1
            else:
                cit_dic[token] += 1
        keys = cit_dic.keys()
        sqrt_cit = 0.0
        for key in keys:
            if key in idf_dic:
                cit_dic[key] = cit_dic[key] / (len(candidate)) * idf_dic[key]
            else:
                cit_dic[key] = 0
            sqrt_cit += cit_dic[key]
        sqrt_cit = math.sqrt(sqrt_cit)
        # 计算相似度
        sum = 0.0
        for key in dic.keys():
            if key in cit_dic:
                sum += dic[key] * cit_dic[key]
        score = sum / (sqrt_source * sqrt_cit)
        scores.append(score)
    new_score = sorted(scores, reverse=True)
    best_indexs = []
    for i in range(num):
        best_indexs.append(scores.index(new_score[i]))
    return best_indexs

def manualselect():
    import random
    datas = pickleload("./data2/random_train_data.pkl", "./data2/random_train_data.pkl")
    select_ids = []
    right_count = 0
    wrong_count = 0
    for _ in range(50):
        id = random.randint(0, len(datas))
        while id in select_ids:
            id = random.randint(0, len(datas)-1)
        select_ids.append(id)

        data = datas[id]
        up_source = data["up_source_tokens"]
        down_source = data["down_source_tokens"]
        target_citation = data['target_tokens']
        citations = data["citations_tokens"]
        print("up_source:",up_source)
        print("down_source:",down_source)

        for index in range(len(citations)):
            if citations[index]['label'] == 1:
                citation = citations[index]["target_tokens"]
                break
        select_lis = [target_citation, citation]

        order = random.randint(0,1)
        if order == 0:
            print(select_lis[0], "\n", select_lis[1])
            inputs = input("输入你要选择的目标：")
            if inputs == "0":
                right_count += 1
            elif inputs == "1":
                wrong_count += 1
        else:
            print(select_lis[1], "\n", select_lis[0])
            inputs = input("输入你要选择的目标：")
            if inputs == "0":
                wrong_count += 1
            elif inputs == "1":
                right_count += 1
    print("right:", right_count)
    print("wrong:", wrong_count)

def getDataDisplay():
    datas = pickleload("./data2/random_train_data.pkl", "./data2/random_train_data.pkl")
    string = "A different strategy is presented in Fung and Chen ( 2004 ) , where English FrameNet entries are mapped to concepts listed in HowNet , an on-line ontology for Chinese , without consulting a parallel corpus . Then , Chinese sentences with predicates instantiating these concepts are found in a monolingual corpus and their arguments are labeled with FrameNet roles . Other work attempts to alleviate the data requirements for semantic role labeling either by relying on unsupervised learning or by extending existing resources through the use of unlabeled data ."
    for i in range(len(datas)):
        if datas[i]['up_source_tokens'] == string:
            citations = datas[i]['citations_tokens']
            print("目标：", datas[i]["target_tokens"])
            for j in range(len(citations)):
                print(j+1, ":")
                print("up:", citations[j]['up_source_tokens'])
                print("down:", citations[j]['down_source_tokens'])
                print(citations[j]['target_tokens'])
                print(citations[j]['label'])
                print("-------------------------------------------------------")

def matplotDataDisplay():
    import matplotlib.pyplot as plt
    datas = pickleload("./data2/random_train_data.pkl", "./data2/random_train_data.pkl")
    key_dic = {}
    for i in range(len(datas)):
        citations = datas[i]['citations_tokens']
        count = 0
        for j in range(len(citations)):
            if citations[j]['label'] == 1:
                count += 1
        if count not in key_dic:
            key_dic[count] = 1
        else:
            key_dic[count] += 1

    new_key_dic = sorted(key_dic.items(), key=lambda item:item[0], reverse=False)
    print(new_key_dic)
    name_list = []
    num_list = []
    for key, value in new_key_dic:
        name_list.append(str(key))
        num_list.append(value)
    plt.bar(range(len(name_list)), num_list, color='grey', tick_label=name_list)
    plt.xlabel('The number of the alternative citations')
    plt.ylabel('Number of the instances')
    plt.show()


def matplotResult():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array([1,2,3,4,5])
    y_Bm25 = np.array([13.79, 18.60, 21.90, 24.48, 26.75])
    y_MatchPyramid  = np.array([13.17, 18.41, 21.97, 24.28, 26.54])
    y_Decomposable  = np.array([16.66, 25.16, 30.64, 34.31, 36.73])
    y_ESIM  = np.array([14.11, 21.78, 26.36, 29.51, 32.01])
    y_IRLM = np.array([22.83, 34.59, 40.21, 44.14, 46.00])
    A,=plt.plot(x, y_IRLM, label='IRLM',linewidth=1.0)
    B,=plt.plot(x, y_Bm25, label='Bm25',linewidth=1.0)
    C,=plt.plot(x, y_MatchPyramid, label='MatchPyramid',linewidth=1.0)
    D,=plt.plot(x, y_Decomposable, label='Decomposable',linewidth=1.0)
    E,=plt.plot(x, y_ESIM, label='ESIM',linewidth=1.0)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 7,
             }
    legend = plt.legend(handles=[A, B,C,D,E], prop=font1)

    plt.xlabel('TopN')
    plt.ylabel('MAP')
    plt.show()

if __name__ == '__main__':
    # getIdf2()
    # manualselect()
    matplotDataDisplay()
    # matplotResult()
    # findBleuSimilar()
    # test_score("./rougetest/data/similar_50_data.json", n_size=4)
    # test_score("./rougetest/data/target_data.json", n_size=1)
    # test_score("./rougetest/data/target_data.json", n_size=2)
    # test_score("./rougetest/data/target_data.json", n_size=3)
    # test_score("./rougetest/data/target_data.json", n_size=4)
    #使用bleu来将候选的citation和真实的citation来对比：
    # sentence = "After each turn , new initiative indices are calculated based on the current indices and the effects of the cues observed during the turn.These cues may be explicit requests by the speaker to give up his initiative , or implicit cues such as ambiguous proposals.The new initiative indices then determine the initiative holders for the next turn ."
    # print(process(sentence))
    pass
