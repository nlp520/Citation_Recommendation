#-*- coding:utf-8 -*-

import logging
from collections import OrderedDict

from Retrieval.data_process import process_kuohao
from Retrieval.models.Lstm_Match import LstmMatch
from Retrieval.models.Transformer import Transformer, Classify, AllClassify, AllClassifyPyramid
from Retrieval.models.arcii import ArcII
from LM_train.prepare import Batch
from Retrieval.train import cal_MAP
from process import pickleload, jsonsave, picklesave
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable

from rougetest.test_bleu import test_bleu
from rougetest.test_rouge import test_rouge
from rougetest.test_score import test_score
from similar import getTopVsmScore

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/dev.log',
                    filemode='w')

def function( xx):
    return 1 if xx != 0 else 0

def test_languageModel(args):
    args.dropout = 0.0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data)*4//5:len(data)]
    # dev_data = data[2000: 4000]

    batch = Batch(args)
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    dev_batches = batch.lm_dev_batch(dev_data, args.context_limit)

    log_msg = "输入词空间大小：%d" %(input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)
    transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))
    if torch.cuda.is_available():
        transform = transform.cuda()

    # 打印参数：
    log_msg = "模型名称：%s \n"%( args.loadmodelName)
    logger.info(log_msg)
    print(log_msg)

    result_dic = {}
    true_label_dic = {}
    all_count = 0
    right_count = 0
    loss_func = torch.nn.NLLLoss()
    loss = 0
    for dev_step, dev_batch in enumerate(dev_batches):
        context_idxs = dev_batch['context_idxs']
        seg_indexs = dev_batch['seg_indexs']
        cit_targets = dev_batch['cit_targets']
        targets = dev_batch['targets']
        target_indexs = dev_batch['target_indexs']
        ref_labels = dev_batch['ref_labels']
        id = dev_batch['id']
        print(id)
        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        context_idxs = Variable(context_idxs).cuda()
        seg_indexs = Variable(seg_indexs).cuda()
        targets = Variable(targets).cuda()
        out1,out2 = transform.forward(context_idxs, seg_indexs, context_mask, target_indexs)
        # print(out)
        for i in range(out1.size(0)):
            loss += loss_func(out1[i], targets[i])
        loss = loss.item() / out1.size(0)
        all_count += 1
        del out1, out2
    print(loss/all_count)

def test(args):
    args.dropout = 0.0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data)*4//5:len(data)]
    # dev_data = data[2000: 4000]

    batch = Batch(args)
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    dev_batches = batch.dev_batch(dev_data, args.context_limit)

    log_msg = "输入词空间大小：%d" %(input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)
    # transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))
    if torch.cuda.is_available():
        transform = transform.cuda()

    # model = Classify(args, transform)
    model = Classify(args, transform)

    #if args.loadmodel ==True:
    model.load_state_dict(torch.load("./modelsave/"+ "maxclassifyModel2.pkl"))

    if torch.cuda.is_available():
        model = model.cuda()

    # 打印参数：
    log_msg = "模型名称：%s \n"%( args.loadmodelName)
    logger.info(log_msg)
    print(log_msg)

    result_dic = {}
    true_label_dic = {}
    for dev_step, dev_batch in enumerate(dev_batches):
        context_idxs = dev_batch['context_idxs']
        seg_indexs = dev_batch['seg_indexs']
        cit_targets = dev_batch['cit_targets']
        target = dev_batch['targets']
        ref_labels = dev_batch['ref_labels']
        id = dev_batch['id']
        print(id)
        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        context_idxs = Variable(context_idxs).cuda()
        seg_indexs = Variable(seg_indexs).cuda()
        out = model.forward(context_idxs, seg_indexs, context_mask)
        # Get loss
        if id not in result_dic:
            result_dic[id] = []
            result_dic[id].append(out.cpu().data)
            true_label_dic[id] = ref_labels
        else:
            result_dic[id].append(out.cpu().data)
        del out
    picklesave(result_dic, "./modelsave/classifyModel2_predict.pkl", "./modelsave/result_dic.pkl")
    picklesave(true_label_dic, "./modelsave/classifyModel2_true.pkl", "./modelsave/true_label_dic.pkl")
    keys = result_dic.keys()
    MAPS = 0
    precisions = 0
    recalls = 0
    for key in keys:
        out = torch.cat(result_dic[key], dim=0)
        predict_index = torch.topk(out, 2, dim=0)[1].squeeze(1).data.numpy()
        # print("预测标签：",predict_index)
        precision, recall, MAP = cal_MAP(true_label_dic[key], predict_index)
        MAPS += MAP
        precisions += precision
        recalls += recall

    MAPS /= len(dev_data)
    precisions /= len(dev_data)
    recalls /= len(dev_data)
    print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))

def all_test(args):
    args.dropout = 0.0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data)*4//5:len(data)]
    # dev_data = data[2000: 4000]

    batch = Batch(args)
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    dev_batches = batch.dev_batch(dev_data, args.context_limit)

    log_msg = "输入词空间大小：%d" %(input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)
    # transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))
    if torch.cuda.is_available():
        transform = transform.cuda()

    # model = Classify(args, transform)
    # model = AllClassify(args, transform)
    model = AllClassifyPyramid(args, transform)

    #if args.loadmodel ==True:
    model.load_state_dict(torch.load("./modelsave/"+ "maxpyramidModel0.pkl"))
    # model.load_state_dict(torch.load("./modelsave/"+ "maxclassifyPyramidModel1.pkl"))

    if torch.cuda.is_available():
        model = model.cuda()

    # 打印参数：
    log_msg = "模型名称：%s \n"%( args.loadmodelName)
    logger.info(log_msg)
    print(log_msg)

    result_dic = {}
    true_label_dic = {}
    for dev_step, dev_batch in enumerate(dev_batches):
        context_idxs = dev_batch['context_idxs']
        source_context_idxs = dev_batch['source_context_idxs']
        seg_indexs = dev_batch['seg_indexs']
        source_seg_indexs = dev_batch['source_seg_indexs']
        cit_targets = dev_batch['cit_targets']
        target = dev_batch['targets']
        id = dev_batch['id']
        print(id)
        if id == 1:
            break
        ref_labels = dev_batch['ref_labels']
        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        source_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in source_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        context_idxs = context_idxs.cuda()
        seg_indexs = seg_indexs.cuda()
        source_context_idxs = source_context_idxs.cuda()
        source_seg_indexs = source_seg_indexs.cuda()
        out = model.forward(context_idxs, seg_indexs, context_mask, source_context_idxs, source_seg_indexs,
                            source_context_mask)

        if id not in result_dic:
            result_dic[id] = []
            result_dic[id].append(out.cpu().data)
            true_label_dic[id] = ref_labels
        else:
            result_dic[id].append(out.cpu().data)
        del out
    # picklesave(result_dic, "./modelsave/pyramidModel2_predict.pkl", "./modelsave/result_dic.pkl")
    # picklesave(true_label_dic, "./modelsave/pyramidModel2_true.pkl", "./modelsave/true_label_dic.pkl")
    keys = result_dic.keys()
    MAPS = 0
    precisions = 0
    recalls = 0
    for key in keys:
        out = torch.cat(result_dic[key], dim=0)
        predict_index = torch.topk(out, 2, dim=0)[1].squeeze(1).data.numpy()
        # print("预测标签：",predict_index)
        precision, recall, MAP = cal_MAP(true_label_dic[key], predict_index)
        MAPS += MAP
        precisions += precision
        recalls += recall

    MAPS /= len(dev_data)
    precisions /= len(dev_data)
    recalls /= len(dev_data)
    print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))

def testMAP():
    result_dic = pickleload("./modelsave/pyramidModel0_predict.pkl", "./modelsave/result_dic.pkl")
    true_label_dic = pickleload("./modelsave/pyramidModel0_true.pkl", "./modelsave/true_label_dic.pkl")
    keys = result_dic.keys()
    MAPS = 0
    precisions = 0
    recalls = 0
    for key in keys:
        out = torch.cat(result_dic[key], dim=0)
        print(out)
        print(true_label_dic[key])
        predict_index = torch.topk(out, 2, dim=0)[1].squeeze(1).data.numpy()
        print("预测标签：", predict_index)
        print("-------------------------------------")
        precision, recall, MAP = cal_MAP(true_label_dic[key], predict_index)
        MAPS += MAP
        precisions += precision
        recalls += recall
    print(len(keys))
    MAPS /= len(keys)
    precisions /= len(keys)
    recalls /= len(keys)
    print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))

def displayData(topn=1):
    result_dic = pickleload("./modelsave/pyramidModel0_predict.pkl", "./modelsave/result_dic.pkl")
    true_label_dic = pickleload("./modelsave/pyramidModel0_true.pkl", "./modelsave/true_label_dic.pkl")
    keys = result_dic.keys()
    blues = 0
    rouges = 0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data) * 4 // 5:len(data)]

    id = 0
    for key in keys:
        up_source_tokens = dev_data[id]["up_source_tokens"]
        target = dev_data[id]["target_tokens"]
        citations = dev_data[id]["citations_tokens"]
        out = torch.cat(result_dic[key], dim=0)
        print("up_source_tokens:",up_source_tokens)
        print("目标target:", target)
        predict_index = torch.topk(out, topn, dim=0)[1].squeeze(1).data.numpy()
        bleu = 0
        rouge = 0
        for index in predict_index:
            alternative_citation = citations[index]["target_tokens"]
            if len(target.strip().split(" ")) < 5 or len(alternative_citation.strip().split(" ")) < 5:
                continue
            bleu += test_bleu(alternative_citation, target, 1)
            rouge += test_rouge(alternative_citation, target)
            print("候选citation：", alternative_citation)
        print("--------------------------")
        bleu = bleu/topn
        rouge = rouge/topn
        blues += bleu
        rouges += rouge
        # print("-----------------------------------------------------")
        id += 1
    blues /= len(keys)
    rouges /= len(keys)
    print("bleu",topn, ":",blues)
    print("rouge:", rouges)

if __name__ == '__main__':
    displayData(3)
    # testMAP()
    pass
