#!usr/bin/python
#-*- coding:utf-8 -*-
import time
import argparse
import os
import torch
from process import prepareData
from Retrieval.train import  doubletrain
from Retrieval.test import test
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
    word2index_pkl = os.path.join("./word2vec", "glove_word2index_300.pkl")#lower_
    source_emb_mat_pkl = os.path.join("./word2vec", "glove_300.pkl")#lower_
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

    parser.add_argument("--glove_word_size", default = int(2.2e6))
    parser.add_argument("--glove_word_dim", default = 300)
    parser.add_argument("--embedding_dim", default = 300)

    #模型相关的参数设置
    parser.add_argument("--type",default = "test", type=str, choices=["prepare", 'doubletrain','train', "test"])
    parser.add_argument("--context_limit",default = 150, type=int)
    parser.add_argument("--test_context_limit", default=150, type=int)
    parser.add_argument("--length", default=150, type=int)
    parser.add_argument("--citation_limit", default=50, type=int)
    parser.add_argument("--num_epoches", default = 20)
    parser.add_argument("--batch_size", default = 16)
    parser.add_argument("--dev_batch_size", default = 1)
    parser.add_argument("--learning_rate", default = 0.0001)
    parser.add_argument("--init_weight_decay", default = 1e-8)
    parser.add_argument("--dropout", default = 0.2) #0.2
    parser.add_argument("--hidden", default = 100)
    parser.add_argument("--hidden_dim", default = 100)
    parser.add_argument("--num_layers", default = 2)
    parser.add_argument("--layer_dim", default = 2)
    parser.add_argument("--bidirectional", default = True)
    parser.add_argument("--padding", default = 0, help="padding")
    parser.add_argument("--modelName", default = "LstmMatch2.pkl")#MatchPyramidModel1
    parser.add_argument("--model", default = "Decomposable", choices=['MatchPyramid', 'LstmMatch','Decomposable','Inference','ESIM','ArcII'])
    parser.add_argument("--optim", default="Adam", choices=["Adadelta", 'Adam', 'SGD'])

    parser.add_argument("--loadmodel", default= True)
    parser.add_argument("--loadmodelName", default= "maxDecomposable.pkl")
    parser.add_argument("--cuda_decices", default= "1")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_decices
    print(args.type)
    if args.type == "prepare": #只预处理一次就行了，多次的话，前面训练的模型，对应的one-hot会变
        prepareData(args)
    # elif args.type == "train":
    #     train(args)
    elif args.type == "doubletrain":
        doubletrain(args)
    elif args.type == "test":
        test(args)

if __name__ == '__main__':
    start = time.time()
    print("start")
    run()
    end = time.time()
    print("end")
    runTime = (end - start)/60.0
    print("程序运行时间是：%.2f"% runTime)
    pass

'''
LstmMatch.pkl





'''


