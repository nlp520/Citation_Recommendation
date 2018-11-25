#-*- coding:utf-8 -*-

import logging
from collections import OrderedDict
import torch.nn.functional as F
from Retrieval.models.Lstm_Match import LstmMatch
from Retrieval.models.Transformer import Transformer, Classify, AllClassify, AllClassifyGetKeyWords, AllClassifyPyramid
from Retrieval.models.arcii import ArcII
from LM_train.prepare import Batch
from Retrieval.train import cal_MAP
from process import pickleload, jsonsave, picklesave
import torch
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable

from rougetest.test_score import test_score
from similar import getTopVsmScore

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/train.log',
                    filemode='w')

def function( xx):
    return 1 if xx != 0 else 0

def train(args):
    train_data = pickleload('../Retrieval/train_data/single_train_data.pkl', "traindata")

    batch = Batch(args)
    # source_embedding = pickleload(args.source_emb_mat_pkl, "source_emb_mat_pkl")
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    train_batches = batch.train_batch(train_data, args.context_limit, args.num_epoches,
                                      args.batch_size)

    log_msg = "输入词空间大小：%d" % (input_vec)
    logger.info(log_msg)
    print(log_msg)

    model = Transformer(args, input_vec)

    if torch.cuda.is_available():
        model = model.cuda()

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)
        param.data.uniform_(-0.08, 0.08)

    parameters_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
                                         weight_decay=args.init_weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)

    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))

    # 打印参数：
    log_msg = "优化函数：%s \n 学习率：%s \n 隐藏层：%s\n 保存模型名称：%s \n" % (
    args.optim, args.learning_rate, args.d_model, args.modelName)
    # print("dropout：", args.dropout)
    logger.info(log_msg)
    print(log_msg)

    set_epoch = 0
    pbar = tqdm(total=len(train_data) * args.num_epoches // args.batch_size + 1)

    loss_func = torch.nn.NLLLoss()
    print_loss_total = 0
    for train_step, (train_batch, epoch) in enumerate(train_batches):
        pbar.update(1)
        context_idxs = train_batch['context_idxs']
        seg_ids = train_batch['seg_indexs']
        target_indexs = train_batch['target_indexs']
        targets = train_batch['targets']
        labels = train_batch['labels']

        # print("up_context_idxs",up_context_idxs)
        # print("down_context_idxs",down_context_idxs)
        # print("target_idxs",target_idxs)
        # print("-----------------------------------------------------")

        context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        context_idxs = Variable(context_idxs).cuda()
        seg_ids = Variable(seg_ids).cuda()
        targets = Variable(targets).cuda()
        labels = Variable(labels).cuda()

        out1, out2 = model.forward(context_idxs, seg_ids, context_mask, target_indexs)
        # Get loss
        optimizer.zero_grad()
        #out1:batch * num_target * word_vec
        #out2:batch * 2
        loss1 = 0
        for i in range(out1.size(0)):
            loss1 += loss_func(out1[i], targets[i])
        loss2 = loss_func(out2, labels)
        loss = loss1 / out1.size(0) + loss2
        # Backward propagation
        loss.backward()
        optimizer.step()
        loss_value = loss.data.item()
        print_loss_total += loss_value

        if train_step%200 == 0:
            log_msg = 'Epoch: %d, Train_step %d  loss: %.4f' % (epoch, train_step, print_loss_total/100)
            logger.debug(log_msg)
            print(log_msg)
            print_loss_total = 0
        if epoch == set_epoch:
            set_epoch += 1
            #实时保存每个epoch的模型
            torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    pbar.close()

def doubletrain(args):
    data = pickleload('../Retrieval/train_data/small_pairs_random_train_data.pkl', "small_pairs_random_train_data")

    dev_data = pickleload("../data2/random_train_data.pkl", "dev_data")
    train_data = data[0] + data[1] + data[2] + data[4]
    dev_data = dev_data[len(dev_data)*3//5:len(dev_data)*4//5]

    batch = Batch(args)
    # source_embedding = pickleload(args.source_emb_mat_pkl, "source_emb_mat_pkl")
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    train_batches = batch.double_train_batch(train_data, args.context_limit, args.num_epoches,
                                      args.batch_size)

    log_msg = "输入词空间大小：%d" % (input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)

    if torch.cuda.is_available():
        transform = transform.cuda()

    transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))

    model = Classify(args, transform)
    model = model.cuda()
    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # for param in model.parameters():
    #     param.data.uniform_(-0.08, 0.08)
    #     param.data.uniform_(-0.08, 0.08)

    parameters_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
                                         weight_decay=args.init_weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)

    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # 打印参数：
    log_msg = "优化函数：%s \n 学习率：%s \n 隐藏层：%s\n 保存模型名称：%s \n" % (
    args.optim, args.learning_rate, args.d_model, args.modelName)
    # print("dropout：", args.dropout)
    logger.info(log_msg)
    print(log_msg)

    set_epoch = 1
    pbar = tqdm(total=len(train_data) * args.num_epoches // args.batch_size + 1)

    def loss_func(high_out , low_out):
        ones = torch.ones(high_out.size(0),1).cuda()
        loss = torch.mean(ones - high_out + low_out)
        return F.relu(loss)
    print_loss_total = 0
    old_accu = 0
    for train_step, (train_batch, epoch) in enumerate(train_batches):
        pbar.update(1)
        high_context_idxs = train_batch['high_cit_context_idxs']
        high_seg_ids = train_batch['high_seg_indexs']
        low_context_idxs = train_batch['low_cit_context_idxs']
        low_seg_ids = train_batch['low_seg_indexs']

        high_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        high_context_idxs = Variable(high_context_idxs).cuda()
        high_seg_ids = Variable(high_seg_ids).cuda()
        low_context_idxs = Variable(low_context_idxs).cuda()
        low_seg_ids = Variable(low_seg_ids).cuda()

        out1 = model.forward(high_context_idxs, high_seg_ids, high_context_mask)
        out2 = model.forward(low_context_idxs, low_seg_ids, low_context_mask)
        # Get loss
        optimizer.zero_grad()
        #out1:batch * num_target * word_vec
        #out2:batch * 2
        loss = loss_func(out1, out2)
        # Backward propagation
        loss.backward()
        optimizer.step()
        loss_value = loss.data.item()
        print_loss_total += loss_value

        if train_step%200 == 0:
            log_msg = 'Epoch: %d, Train_step %d  loss: %.4f' % (epoch, train_step, print_loss_total/200)
            logger.debug(log_msg)
            print(log_msg)
            print_loss_total = 0
        if epoch == set_epoch:
            set_epoch += 1
            dev_batches = batch.dev_batch(dev_data, args.context_limit)
            result_dic = {}
            true_label_dic = {}
            for dev_step, dev_batch in enumerate(dev_batches):
                context_idxs = dev_batch['context_idxs']
                seg_indexs = dev_batch['seg_indexs']
                cit_targets = dev_batch['cit_targets']
                target = dev_batch['targets']
                ref_labels = dev_batch['ref_labels']
                id = dev_batch['id']
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
            picklesave(result_dic, "./modelsave/classify_dev_result_dic.pkl", "./modelsave/result_dic.pkl")
            picklesave(true_label_dic, "./modelsave/classify_dev_true_label_dic.pkl", "./modelsave/true_label_dic.pkl")
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
            all_loss = MAPS
            if all_loss > old_accu:
                old_accu = all_loss
                torch.save(model.state_dict(), "./modelsave/max" + args.modelName)
                best_epoch = epoch
            # else:
            #     args.learning_rate = args.learning_rate / 2.0
            #     if args.learning_rate <= 1e-6:
            #         args.learning_rate = 1e-6
            #     if args.optim == "Adadelta":
            #         optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
            #                                          weight_decay=args.init_weight_decay)
            #     elif args.optim == "Adam":
            #         optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate,
            #                                      weight_decay=args.init_weight_decay)
            #     elif args.optim == "SGD":
            #         optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate,
            #                                     weight_decay=args.init_weight_decay)
            log_msg = '\n验证集的bleu为: %.4f\n 取得最小loss的epoch为：%d' % (all_loss, best_epoch)
            logger.info(log_msg)
            print(log_msg)
            # 实时保存每个epoch的模型
            torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    pbar.close()

def all_doubletrain(args):
    data = pickleload('../Retrieval/train_data/small_pairs_random_train_data.pkl', "small_pairs_train_data")
    dev_data = pickleload("../data2/random_train_data.pkl", "dev_data")
    train_data = data[0] + data[1] + data[2] + data[3]
    dev_data = dev_data[len(dev_data)*4//5:len(dev_data)]

    batch = Batch(args)
    # source_embedding = pickleload(args.source_emb_mat_pkl, "source_emb_mat_pkl")
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    train_batches = batch.double_train_batch(train_data, args.context_limit, args.num_epoches,
                                      args.batch_size)

    log_msg = "输入词空间大小：%d" % (input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)

    if torch.cuda.is_available():
        transform = transform.cuda()

    transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))

    # model = AllClassify(args, transform)
    model = AllClassifyPyramid(args, transform)

    model = model.cuda()
    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # for param in model.parameters():
    #     param.data.uniform_(-0.08, 0.08)
    #     param.data.uniform_(-0.08, 0.08)

    parameters_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
                                         weight_decay=args.init_weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)

    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # 打印参数：
    log_msg = "优化函数：%s \n 学习率：%s \n 隐藏层：%s\n 保存模型名称：%s \n" % (
    args.optim, args.learning_rate, args.d_model, args.modelName)
    # print("dropout：", args.dropout)
    logger.info(log_msg)
    print(log_msg)

    set_epoch = 1
    pbar = tqdm(total=len(train_data) * args.num_epoches // args.batch_size + 1)

    def loss_func(high_out , low_out):
        ones = torch.ones(high_out.size(0),1).cuda()
        loss = torch.mean(ones - high_out + low_out)
        return F.relu(loss)
    print_loss_total = 0
    old_accu = 0
    for train_step, (train_batch, epoch) in enumerate(train_batches):
        pbar.update(1)
        high_context_idxs = train_batch['high_cit_context_idxs']
        high_seg_ids = train_batch['high_seg_indexs']
        low_context_idxs = train_batch['low_cit_context_idxs']
        low_seg_ids = train_batch['low_seg_indexs']
        high_source_context_idxs = train_batch['high_source_context_idxs']
        high_source_seg_indexs = train_batch['high_source_seg_indexs']
        low_source_context_idxs = train_batch['low_source_context_idxs']
        low_source_seg_indexs = train_batch['low_source_seg_indexs']

        high_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        high_source_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_source_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_source_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_source_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        high_context_idxs = Variable(high_context_idxs).cuda()
        high_seg_ids = Variable(high_seg_ids).cuda()
        low_context_idxs = Variable(low_context_idxs).cuda()
        low_seg_ids = Variable(low_seg_ids).cuda()
        high_source_context_idxs = Variable(high_source_context_idxs).cuda()
        high_source_seg_indexs = Variable(high_source_seg_indexs).cuda()
        low_source_context_idxs = Variable(low_source_context_idxs).cuda()
        low_source_seg_indexs = Variable(low_source_seg_indexs).cuda()

        out1 = model.forward(high_context_idxs, high_seg_ids, high_context_mask, high_source_context_idxs, high_source_seg_indexs, high_source_context_mask)
        out2 = model.forward(low_context_idxs, low_seg_ids, low_context_mask, low_source_context_idxs, low_source_seg_indexs, low_source_context_mask)
        # Get loss
        optimizer.zero_grad()
        #out1:batch * num_target * word_vec
        #out2:batch * 2
        loss = loss_func(out1, out2)
        # Backward propagation
        loss.backward()
        optimizer.step()
        loss_value = loss.data.item()
        print_loss_total += loss_value
        del out1, out2
        if train_step%100 == 0:
            log_msg = 'Epoch: %d, Train_step %d  loss: %.4f' % (epoch, train_step, print_loss_total/100)
            logger.debug(log_msg)
            print(log_msg)
            print_loss_total = 0
        if epoch == set_epoch:
            set_epoch += 1
            dev_batches = batch.dev_batch(dev_data, args.context_limit)
            result_dic = {}
            true_label_dic = {}
            for dev_step, dev_batch in enumerate(dev_batches):
                context_idxs = dev_batch['context_idxs']
                source_context_idxs = dev_batch['source_context_idxs']
                seg_indexs = dev_batch['seg_indexs']
                source_seg_indexs = dev_batch['source_seg_indexs']
                ref_labels = dev_batch['ref_labels']
                id = dev_batch['id']

                context_mask = torch.Tensor(
                    np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                             dtype=np.float)).cuda()
                source_context_mask = torch.Tensor(
                    np.array([list(map(function, xx)) for xx in source_context_idxs.data.numpy()],
                             dtype=np.float)).cuda()

                context_idxs = Variable(context_idxs).cuda()
                seg_indexs = Variable(seg_indexs).cuda()
                source_context_idxs = Variable(source_context_idxs).cuda()
                source_seg_indexs = Variable(source_seg_indexs).cuda()
                out = model.forward(context_idxs, seg_indexs, context_mask, source_context_idxs,source_seg_indexs ,source_context_mask)
                # Get loss
                if id not in result_dic:
                    result_dic[id] = []
                    result_dic[id].append(out.cpu().data)
                    true_label_dic[id] = ref_labels
                else:
                    result_dic[id].append(out.cpu().data)
                del out
            picklesave(result_dic, "./modelsave/all_dev_result_dic22.pkl", "./modelsave/result_dic.pkl")
            picklesave(true_label_dic, "./modelsave/all_dev_true_label_dic22.pkl", "./modelsave/true_label_dic.pkl")
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
            all_loss = MAPS
            if all_loss > old_accu:
                old_accu = all_loss
                torch.save(model.state_dict(), "./modelsave/max" + args.modelName)
                best_epoch = epoch
            # else:
            #     args.learning_rate = args.learning_rate / 2.0
            #     if args.learning_rate <= 1e-6:
            #         args.learning_rate = 1e-6
            #     if args.optim == "Adadelta":
            #         optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
            #                                          weight_decay=args.init_weight_decay)
            #     elif args.optim == "Adam":
            #         optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate,
            #                                      weight_decay=args.init_weight_decay)
            #     elif args.optim == "SGD":
            #         optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate,
            #                                     weight_decay=args.init_weight_decay)
            log_msg = '\n验证集的MAP为: %.4f  P为: %.4f  R为: %.4f\n 取得最小loss的epoch为：%d' % (
                    all_loss, precisions, recalls, best_epoch)
            logger.info(log_msg)
            print(log_msg)
            # 实时保存每个epoch的模型
            torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    pbar.close()


def all_doubletrainKey(args):
    data = pickleload('../Retrieval/train_data/small_pairs_random_train_data.pkl', "small_pairs_random_train_data")
    dev_data = pickleload("../data2/random_train_data.pkl", "dev_data")
    train_data = data[0] + data[1] + data[2] + data[3]
    dev_data = dev_data[len(dev_data)*4//5:len(dev_data)]

    batch = Batch(args)
    # source_embedding = pickleload(args.source_emb_mat_pkl, "source_emb_mat_pkl")
    word2index = pickleload("./word_vec/word2index.pkl", "word2index.pkl")
    input_vec = len(word2index)

    train_batches = batch.double_train_batch(train_data, args.context_limit, args.num_epoches,
                                      args.batch_size)

    log_msg = "输入词空间大小：%d" % (input_vec)
    logger.info(log_msg)
    print(log_msg)

    transform = Transformer(args, input_vec)

    if torch.cuda.is_available():
        transform = transform.cuda()

    transform.load_state_dict(torch.load("./modelsave/" + "TransformModel0.pkl"))

    model = AllClassifyGetKeyWords(args, transform)

    model = model.cuda()
    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # for param in model.parameters():
    #     param.data.uniform_(-0.08, 0.08)
    #     param.data.uniform_(-0.08, 0.08)

    parameters_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
                                         weight_decay=args.init_weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate, weight_decay=args.init_weight_decay)

    if args.loadmodel == True:
        model.load_state_dict(torch.load("./modelsave/" + args.loadmodelName))
    # 打印参数：
    log_msg = "优化函数：%s \n 学习率：%s \n 隐藏层：%s\n 保存模型名称：%s \n" % (
    args.optim, args.learning_rate, args.d_model, args.modelName)
    # print("dropout：", args.dropout)
    logger.info(log_msg)
    print(log_msg)

    set_epoch = 1
    pbar = tqdm(total=len(train_data) * args.num_epoches // args.batch_size + 1)

    def loss_func(high_out , low_out, seleout11, seleout12, seleout21, seleout22):
        ones = torch.ones(high_out.size(0),1).cuda()
        ones1 = 7*torch.ones(high_out.size(0),1).cuda()
        loss = torch.mean(ones - high_out + low_out) + torch.mean((ones1 - seleout11)*(ones1 - seleout11)) + torch.mean((ones1 - seleout12)*(ones1 - seleout12)) + \
               torch.mean((ones1 - seleout21)*(ones1 - seleout21)) + torch.mean((ones1 - seleout22)*(ones1 - seleout22))
        return F.relu(loss), torch.mean(ones - high_out + low_out)
    print_loss_total = 0
    old_accu = 0
    print_loss_total2= 0
    for train_step, (train_batch, epoch) in enumerate(train_batches):
        pbar.update(1)
        high_context_idxs = train_batch['high_cit_context_idxs']
        high_seg_ids = train_batch['high_seg_indexs']
        low_context_idxs = train_batch['low_cit_context_idxs']
        low_seg_ids = train_batch['low_seg_indexs']
        high_source_context_idxs = train_batch['high_source_context_idxs']
        high_source_seg_indexs = train_batch['high_source_seg_indexs']
        low_source_context_idxs = train_batch['low_source_context_idxs']
        low_source_seg_indexs = train_batch['low_source_seg_indexs']

        high_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        high_source_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in high_source_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()
        low_source_context_mask = torch.Tensor(
            np.array([list(map(function, xx)) for xx in low_source_context_idxs.data.numpy()],
                     dtype=np.float)).cuda()

        high_context_idxs = Variable(high_context_idxs).cuda()
        high_seg_ids = Variable(high_seg_ids).cuda()
        low_context_idxs = Variable(low_context_idxs).cuda()
        low_seg_ids = Variable(low_seg_ids).cuda()
        high_source_context_idxs = Variable(high_source_context_idxs).cuda()
        high_source_seg_indexs = Variable(high_source_seg_indexs).cuda()
        low_source_context_idxs = Variable(low_source_context_idxs).cuda()
        low_source_seg_indexs = Variable(low_source_seg_indexs).cuda()

        out1, seleout11, seleout12 = model.forward(high_context_idxs, high_seg_ids, high_context_mask, high_source_context_idxs, high_source_seg_indexs, high_source_context_mask)
        out2, seleout21, seleout22= model.forward(low_context_idxs, low_seg_ids, low_context_mask, low_source_context_idxs, low_source_seg_indexs, low_source_context_mask)
        # Get loss
        optimizer.zero_grad()
        #out1:batch * num_target * word_vec
        #out2:batch * 2
        loss, loss2 = loss_func(out1, out2, seleout11, seleout12, seleout21, seleout22)
        # Backward propagation
        loss.backward()
        optimizer.step()
        loss_value = loss.data.item()
        print_loss_total += loss_value
        print_loss_total2 += loss2.data.item()
        del out1, out2
        if train_step%100 == 0:
            log_msg = 'Epoch: %d, Train_step %d  loss1: %.4f, loss2:%.4f' % (epoch, train_step, print_loss_total/100, print_loss_total2/100)
            logger.debug(log_msg)
            print(log_msg)
            print_loss_total = 0
            print_loss_total2 = 0
        if epoch == set_epoch:
            set_epoch += 1
            dev_batches = batch.dev_batch(dev_data, args.context_limit)
            result_dic = {}
            true_label_dic = {}
            for dev_step, dev_batch in enumerate(dev_batches):
                context_idxs = dev_batch['context_idxs']
                source_context_idxs = dev_batch['source_context_idxs']
                seg_indexs = dev_batch['seg_indexs']
                source_seg_indexs = dev_batch['source_seg_indexs']
                ref_labels = dev_batch['ref_labels']
                id = dev_batch['id']

                context_mask = torch.Tensor(
                    np.array([list(map(function, xx)) for xx in context_idxs.data.numpy()],
                             dtype=np.float)).cuda()
                source_context_mask = torch.Tensor(
                    np.array([list(map(function, xx)) for xx in source_context_idxs.data.numpy()],
                             dtype=np.float)).cuda()

                context_idxs = Variable(context_idxs).cuda()
                seg_indexs = Variable(seg_indexs).cuda()
                source_context_idxs = Variable(source_context_idxs).cuda()
                source_seg_indexs = Variable(source_seg_indexs).cuda()
                out, seleout1, seleout2 = model.forward(context_idxs, seg_indexs, context_mask, source_context_idxs,source_seg_indexs ,source_context_mask)
                # Get loss
                if id not in result_dic:
                    result_dic[id] = []
                    result_dic[id].append(out.cpu().data)
                    true_label_dic[id] = ref_labels
                else:
                    result_dic[id].append(out.cpu().data)
                del out
            picklesave(result_dic, "./modelsave/all_dev_result_dic22.pkl", "./modelsave/result_dic.pkl")
            picklesave(true_label_dic, "./modelsave/all_dev_true_label_dic22.pkl", "./modelsave/true_label_dic.pkl")
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
            all_loss = MAPS
            if all_loss > old_accu:
                old_accu = all_loss
                torch.save(model.state_dict(), "./modelsave/max" + args.modelName)
                best_epoch = epoch
            # else:
            #     args.learning_rate = args.learning_rate / 2.0
            #     if args.learning_rate <= 1e-6:
            #         args.learning_rate = 1e-6
            #     if args.optim == "Adadelta":
            #         optimizer = torch.optim.Adadelta(parameters_trainable, lr=args.learning_rate,
            #                                          weight_decay=args.init_weight_decay)
            #     elif args.optim == "Adam":
            #         optimizer = torch.optim.Adam(parameters_trainable, lr=args.learning_rate,
            #                                      weight_decay=args.init_weight_decay)
            #     elif args.optim == "SGD":
            #         optimizer = torch.optim.SGD(parameters_trainable, lr=args.learning_rate,
            #                                     weight_decay=args.init_weight_decay)
            log_msg = '\n验证集的MAP为: %.4f  P为: %.4f  R为: %.4f\n 取得最小loss的epoch为：%d' % (
                    all_loss, precisions, recalls, best_epoch)
            logger.info(log_msg)
            print(log_msg)
            # 实时保存每个epoch的模型
            torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    torch.save(model.state_dict(), "./modelsave/" + args.modelName)
    pbar.close()
