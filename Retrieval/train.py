#-*- coding:utf-8 -*-

import logging
from collections import OrderedDict
import torch.nn.functional as F

from Retrieval.models.Decomposable import Decomposable
from Retrieval.models.ESIM import ESIM
from Retrieval.models.Inference import Inference
from Retrieval.models.Lstm_Match import LstmMatch
from Retrieval.models.arcii import ArcII
from Retrieval.models.matchPyramid import MatchPyramid
from Retrieval.prepare import Batch
from process import pickleload, jsonsave
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/train.log',
                    filemode='w')

def function( xx):
    return 1 if xx != 0 else 0

def doubletrain(args):

    '''
    两个citation  high和low一起进行训练
    :param args:
    :return:
    '''
    data = pickleload("./train_data/small_pairs_random_train_data.pkl", "small_pairs_random_train_data")
    dev_data = pickleload("../data2/random_train_data.pkl", "dev_data")
    train_data = data[0] + data[1] + data[2] + data[3]
    dev_data = dev_data[len(dev_data)*4//5:len(dev_data)]

    batch = Batch(args)
    word2index = pickleload(args.word2index_pkl, 'word2index')
    input_vec = len(word2index)
    # source_embedding = pickleload(args.source_emb_mat_pkl, "source_emb_mat_pkl")
    source_embedding = pickleload("./word2vec/glove_300.pkl", "glove_300.pkl")
    source_embedding = np.array(source_embedding, dtype=np.float32)

    train_batches = batch.double_train_batch(train_data, args.context_limit, args.citation_limit, args.num_epoches, args.batch_size)

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



    if torch.cuda.is_available():
        model = model.cuda()

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)
        param.data.uniform_(-0.08, 0.08)

    parameters_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)

    if args.loadmodel ==True:
        model.load_state_dict(torch.load("./modelsave/"+args.loadmodelName))

    # 打印参数：
    log_msg = "优化函数：%s \n 学习率：%s \n 隐藏层：%s\n 保存模型名称：%s \n"%(args.optim, args.learning_rate, args.hidden, args.modelName)
    print("dropout：", args.dropout)
    logger.info(log_msg)
    print(log_msg)

    set_epoch = 1
    pbar = tqdm(total=len(train_data) * args.num_epoches // args.batch_size + 1)
    best_epoch = 0
    old_accu = 0
    def loss_func(high_out , low_out):
        ones = torch.ones(high_out.size(0),1).cuda()
        loss = torch.mean(ones - high_out + low_out)
        return F.relu(loss)
    print_loss_total = 0
    for train_step, (train_batch, epoch) in enumerate(train_batches):
        pbar.update(1)
        context_idxs = train_batch['context_idxs']
        high_cit_context_idxs = train_batch['high_cit_context_idxs']
        low_cit_context_idxs = train_batch['low_cit_context_idxs']

        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        high_cit_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_cit_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_cit_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_cit_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        context_idxs = Variable(context_idxs).cuda()
        high_cit_context_idxs = Variable(high_cit_context_idxs).cuda()
        low_cit_context_idxs = Variable(low_cit_context_idxs).cuda()

        high_out = model.forward(context_idxs, high_cit_context_idxs, context_mask, high_cit_context_mask)
        low_out = model.forward(context_idxs, low_cit_context_idxs, context_mask, low_cit_context_mask)
        # Get loss
        optimizer.zero_grad()
        # print("high_out:",high_out.size())
        # print("low_out:",low_out.size())
        loss = loss_func(high_out, low_out)
        # Backward propagation
        loss.backward()
        optimizer.step()
        loss_value = loss.data.item()
        print_loss_total += loss_value

        if train_step%100 == 0:
            log_msg = 'Epoch: %d, Train_step %d  loss: %.4f' % (epoch, train_step, print_loss_total/100)
            logger.debug(log_msg)
            print(log_msg)
            print_loss_total = 0
        if epoch == set_epoch:
            set_epoch += 1
            dev_batches = batch.dev_batch(dev_data, args.context_limit, args.citation_limit)
            pbar2 = tqdm(total=len(dev_data))
            MAPS = 0
            precisions = 0
            recalls = 0
            for dev_step, dev_batch  in enumerate(dev_batches):
                pbar2.update(1)
                context_idxs = dev_batch['context_idxs']
                cit_context_idxs = dev_batch['cit_context_idxs']
                ref_labels = dev_batch['ref_labels']
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
                predict_index = torch.topk(out, 3, dim=0)[1].squeeze(1).data.cpu().numpy()

                precision, recall, MAP = cal_MAP(ref_labels, predict_index)
                MAPS += MAP
                precisions += precision
                recalls += recall

            MAPS /= len(dev_data)
            precisions /= len(dev_data)
            recalls /= len(dev_data)
            all_loss = MAPS
            pbar2.close()

            if all_loss > old_accu:
                old_accu = all_loss
                torch.save(model.state_dict(), "./modelsave/max" + args.modelName)
                best_epoch = epoch
            else:
                args.learning_rate = args.learning_rate/2.0
                if args.learning_rate <= 1e-6:
                    args.learning_rate = 1e-6
                if args.optim == "Adadelta":
                    optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
                                                     weight_decay=args.init_weight_decay)
                elif args.optim == "Adam":
                    optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate,
                                                 weight_decay=args.init_weight_decay)
                elif args.optim == "SGD":
                    optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate,
                                                weight_decay=args.init_weight_decay)
            log_msg = '\n验证集的MAP为: %.4f  P为: %.4f  R为: %.4f\n 取得最小loss的epoch为：%d' % (all_loss, precisions, recalls , best_epoch)
            logger.info(log_msg)
            print(log_msg)
            #实时保存每个epoch的模型
            torch.save(model.state_dict(), "./modelsave/" + args.modelName)

    torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    pbar.close()


def cal_MAP(ref_lis, pre_lis):
    '''
    计算单个检索的MAP， Precision, Recall
    :param ref_lis:
    :param pre_lis:
    :return:
    '''
    if len(ref_lis) == 0:
        return 0, 0, 0
    MAP = 0
    true_count = 0
    for i in pre_lis:
        if i in ref_lis:
            true_count += 1
    precision = true_count/len(pre_lis)
    recall = true_count/len(ref_lis)

    count = 1
    for i in range(len(pre_lis)):
        if pre_lis[i] in ref_lis:
            MAP += count/(i+1)
            count += 1
    MAP = MAP / len(ref_lis)
    return precision, recall, MAP


if __name__ == '__main__':

    pass






