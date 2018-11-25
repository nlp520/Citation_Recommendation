#-*- coding:utf-8 -*-

import logging
from collections import OrderedDict

from Retrieval.models.Decomposable import Decomposable
from Retrieval.models.ESIM import ESIM
from Retrieval.models.Inference import Inference
from Retrieval.models.Lstm_Match import LstmMatch
from Retrieval.models.arcii import ArcII
from Retrieval.models.matchPyramid import MatchPyramid
from Retrieval.prepare import Batch
from Retrieval.train import cal_MAP

from process import pickleload, jsonsave
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable

from rougetest.test_bleu import test_bleu
from rougetest.test_rouge import test_rouge
from rougetest.test_score import test_score

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/dev.log',
                    filemode='w')

def function( xx):
    return 1 if xx != 0 else 0

def test(args):
    args.dropout = 0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data)*4//5:len(data)]
    # dev_data = data[2000: 4000]

    batch = Batch(args)
    word2index = pickleload(args.word2index_pkl, 'word2index')
    input_vec = len(word2index)
    source_embedding = pickleload("./word2vec/glove_300.pkl", "glove_300.pkl")
    source_embedding = np.array(source_embedding, dtype=np.float32)

    dev_batches = batch.dev_batch(dev_data, args.context_limit, args.citation_limit)

    log_msg = "输入词空间大小：%d" %(input_vec)
    logger.info(log_msg)
    print(log_msg)

    if args.model == "MatchPyramid":
        model = MatchPyramid(args, input_vec, source_embedding)
    elif args.model == "LstmMatch":
        model = LstmMatch(args, input_vec, source_embedding)
    elif args.model == "Decomposable":
        model = Decomposable(args, input_vec, source_embedding)
    elif args.model == "Inference":
        model = Inference(args, input_vec, source_embedding)
    elif args.model == "ESIM":
        model = ESIM(args, input_vec, source_embedding)
    elif args.model == "ArcII":
        model = ArcII(args, input_vec, source_embedding)

    if args.loadmodel ==True:
        model.load_state_dict(torch.load("./modelsave/"+ args.loadmodelName))

    if torch.cuda.is_available():
        model = model.cuda()

    # 打印参数：
    log_msg = "模型名称：%s \n"%( args.loadmodelName)
    logger.info(log_msg)
    print(log_msg)

    pbar2 = tqdm(total=len(dev_data))
    MAPS = 0
    precisions = 0
    recalls = 0
    blues = 0
    rouges = 0
    for dev_step, dev_batch in enumerate(dev_batches):
        pbar2.update(1)
        context_idxs = dev_batch['context_idxs']
        cit_context_idxs = dev_batch['cit_context_idxs']
        ref_labels = dev_batch['ref_labels']
        target = dev_batch["targets"]
        citations = dev_batch['citations']
        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        cit_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in cit_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        context_idxs = Variable(context_idxs).cuda()
        cit_context_idxs = Variable(cit_context_idxs).cuda()

        out = model.forward(context_idxs, cit_context_idxs, context_mask, cit_context_mask)
        # Get loss
        # print("真实值：",out)
        # print("真实标签：",ref_labels)
        topn = 3
        predict_index = torch.topk(out,topn, dim=0)[1].squeeze(1).data.cpu().numpy()

        bleu = 0
        rouge = 0

        for index in predict_index:
            alternative_citation = citations[index]["target_tokens"]
            bleu += test_bleu(alternative_citation, target, 1)
            rouge += test_rouge(alternative_citation, target)
            # print("候选citation：", alternative_citation)
        bleu = bleu / topn
        rouge = rouge / topn
        blues += bleu
        rouges += rouge

        # print("预测标签：",predict_index)
        precision, recall, MAP = cal_MAP(ref_labels, predict_index)
        MAPS += MAP
        precisions += precision
        recalls += recall

    MAPS /= len(dev_data)
    precisions /= len(dev_data)
    recalls /= len(dev_data)
    blues /= len(dev_data)
    rouges /= len(dev_data)
    print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))
    print("bleu", topn, ":", blues)
    print("rouge:", rouges)
    pbar2.close()

def testRetrievalModelResult(topn = 5):
    with open("./result/predict.test.arcii_ranking.txt") as fp:
        lines = fp.readlines()
    result_lis = {}
    last_name = ""
    for line in lines:
        results = line.replace("\n","").split("	")
        if last_name != results[0]:
            last_name = results[0]
            result_lis[last_name] = []
        result_lis[last_name].append(int(results[2].split("_")[-1]))
    predict_indexs = [value for key, value in result_lis.items()]
    MAPS = 0
    precisions = 0
    recalls = 0
    data = pickleload("../data2/random_train_data.pkl", "traindata")
    dev_data = data[len(data) * 4 // 5:len(data)]
    bleus = 0
    rouges = 0
    for id in range(len(dev_data)):
        target = dev_data[id]["target_tokens"]
        citations = dev_data[id]['citations_tokens']
        true_label = []
        predict_index = predict_indexs[id][0:topn]
        for i in range(len(citations)):
            citation = citations[i]
            if citation['label'] == 1:
                true_label.append(i)
        bleu = 0
        rouge = 0
        for predict in predict_index:
            # print(predict)
            alternative_citation = citations[predict]["target_tokens"]
            if len(target.strip().split(" ")) < 5 or len(alternative_citation.strip().split(" ")) < 5:
                continue
            bleu += test_bleu(alternative_citation, target, 1)
            rouge += test_rouge(alternative_citation, target)
            print(bleu)
        print("------------------------")
        bleus += bleu/len(predict_index)
        rouges += rouge/len(predict_index)
        precision, recall, MAP = cal_MAP(true_label, predict_index)
        precisions += precision
        recalls+= recall
        MAPS += MAP
    MAPS /= len(predict_indexs)
    precisions /= len(predict_indexs)
    recalls /= len(predict_indexs)
    bleus /= len(dev_data)
    rouges /= len(dev_data)
    print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))
    print("bleu", topn, ":", bleus)
    print("rouge:", rouges)

if __name__ == '__main__':
    testRetrievalModelResult(topn=3)

    pass

