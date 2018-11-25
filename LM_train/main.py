#!usr/bin/python
#-*- coding:utf-8 -*-
import time
import argparse
import os
import torch
from process import prepareData
from LM_train.train import train, doubletrain, all_doubletrain, all_doubletrainKey
from LM_train.test import test, all_test, test_languageModel

'''

'''
def run():
    torch.manual_seed(5000)
    data_home = "../data"
    glove_word_txt = os.path.join("../glove", "glove.840B.300d.txt")

    goldendata = os.path.join(data_home, "citationGoldenData.pkl")
    traindata = os.path.join("./train_data", "single_train_data.pkl")
    targetindex2word_pkl = os.path.join("../glove", "lower_target_index2word.pkl")
    target_word2index_pkl = os.path.join("../glove", "lower_target_word2index.pkl")
    word2index_pkl = os.path.join("./word2vec", "word2vec_word2index_300.pkl")#lower_
    source_emb_mat_pkl = os.path.join("./word2vec", "embedding_vec_300.pkl")#lower_
    target_emb_mat_pkl = os.path.join("../glove", "embedding_vec_300.pkl")#lower_

    parser = argparse.ArgumentParser()
    #数据文件路径
    parser.add_argument("--goldendata", default=goldendata)
    parser.add_argument("--traindata", default=traindata)
    parser.add_argument("--glove_word_txt", default=glove_word_txt)
    parser.add_argument("--targetindex2word_pkl", default=targetindex2word_pkl)
    parser.add_argument("--target_word2index_pkl", default=target_word2index_pkl)
    parser.add_argument("--target_emb_mat_pkl", default=target_emb_mat_pkl)
    parser.add_argument("--source_emb_mat_pkl", default=source_emb_mat_pkl)
    parser.add_argument("--word2index_pkl", default=word2index_pkl)

    #模型相关的参数设置
    parser.add_argument("--type",default = "all_test", type=str, choices=["test_languageModel","prepare", 'doubletrain','train', "test", 'all_doubletrain','all_test', "all_doubletrainKey"])
    parser.add_argument("--context_limit",default = 250, type=int)
    parser.add_argument("--test_context_limit", default=250, type=int)
    parser.add_argument("--length", default = 250, type=int)
    parser.add_argument("--citation_limit", default=30, type=int)
    parser.add_argument("--num_epoches", default = 10)
    parser.add_argument("--batch_size", default = 8)
    parser.add_argument("--dev_batch_size", default = 1)
    parser.add_argument("--learning_rate", default = 0.000025)
    parser.add_argument("--init_weight_decay", default = 1e-8)
    parser.add_argument("--dropout", default = 0.2) # 0.2
    parser.add_argument("--modelName", default = "pyramidModel2.pkl")#all_classifyModel2    pyramidModel1  classifyModel1
    parser.add_argument("--model", default = "Transform", choices=['ArcII', 'LstmMatch', 'Transform'])
    parser.add_argument("--optim", default="Adam", choices=["Adadelta", 'Adam', 'SGD'])

    parser.add_argument("--loadmodel", default= True)
    parser.add_argument("--loadmodelName", default= "maxpyramidModel2.pkl")
    parser.add_argument("--cuda_decices", default= "1")

    #transform 参数：
    parser.add_argument('--d_model', default=300)
    parser.add_argument('--d_inner_hid', default=1024)
    parser.add_argument('--n_layers', default=6)
    parser.add_argument('--n_head', default=8)
    parser.add_argument('--d_k', default=60)
    parser.add_argument('--d_v', default=60)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_decices
    print(args.type)
    if args.type == "train":
        train(args)
    elif args.type == "test":
        test(args)
    elif args.type == "all_test":
        all_test(args)
    elif args.type == "doubletrain":
        doubletrain(args)
    elif args.type == "all_doubletrain":
        all_doubletrain(args)
    elif args.type == "all_doubletrainKey":
        all_doubletrainKey(args)
    elif args.type == "test_languageModel":
        test_languageModel(args)

if __name__ == '__main__':
    start = time.time()
    print("start")
    run()
    end = time.time()
    print("end")
    runTime = (end - start)/60.0
    print("程序运行时间是：%.2f"% runTime)
    pass
