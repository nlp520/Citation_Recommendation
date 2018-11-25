#-*- coding:utf-8 -*-
import pickle

from process import pickleload
import numpy as np
import torch

from similar import getTopSimilar
import random

class Batch():
    def __init__(self, args):
        self.OOV = 0
        self.OOD = 1 #unknow
        self.word2index = pickleload('./word_vec/word2index.pkl', "word2index")
        self.idf_dic = pickleload("../data2/idf.pkl", "idf.pkl")

    def getindex(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.OOD

    def process(self, up_sources, down_sources, cit_up_sources, context_limit):
        content_source = ['CLS'] + up_sources.split(" ") + ['DSP'] + cit_up_sources.split(" ") + ['DSP'] + down_sources.split(" ") + ['DSP']

        while ' 'in content_source:
            content_source.remove(" ")
        context_idxs = np.zeros([context_limit], dtype=np.int64)
        seg_indx = np.zeros([context_limit], dtype=np.int64)

        min_context = len(content_source) if len(content_source) < context_limit else context_limit
        target_index = []
        target = []
        flag = 1

        rand_lis = []
        while(len(rand_lis) != 8):
            x = random.randint(1, min_context-1)
            if x not in rand_lis:
                rand_lis.append(x)

        for i in range(0, min_context):
            seg_indx[i] = flag
            if content_source[i] == "<DSP>":
                flag += 1

            if i not in rand_lis:
                context_idxs[i] = self.getindex(content_source[i])
            else:
                target_index.append(i)
                target.append(self.getindex(content_source[i]))
                rand_int = random.randint(0,9)
                if rand_int < 8:
                    context_idxs[i] = self.getindex("<MASK>")
                elif rand_int == 8:
                    context_idxs[i] = self.getindex(content_source[i])
                else:
                    context_idxs[i] = self.getindex(random.randint(2, len(self.word2index)-3))

        return context_idxs, target_index, target, seg_indx

    def train_process(self, up_sources, down_sources, cit_up_sources, context_limit):
        content_source = ['CLS'] + up_sources.split(" ") + ['DSP'] + cit_up_sources.split(" ") + ['DSP'] + down_sources.split(" ") + ['DSP']

        while ' 'in content_source:
            content_source.remove(" ")
        context_idxs = np.zeros([context_limit], dtype=np.int64)
        seg_indx = np.zeros([context_limit], dtype=np.int64)

        min_context = len(content_source) if len(content_source) < context_limit else context_limit
        flag = 1

        for i in range(0, min_context):
            seg_indx[i] = flag
            if content_source[i] == "<DSP>":
                flag += 1
            context_idxs[i] = self.getindex(content_source[i])
        return context_idxs, seg_indx

    def train_batch(self, data, context_limit, num_epoches, batch_size):
        '''
        [
            {
             "up_source_tokens":"",
             "down_source_tokens":"",
             "target_tokens":""
             "cit_up_source_tokens":"",
             "cit_down_source_tokens":"",
             "cit_target_tokens":""
            }
        ]
        :param data:
        :param context_limit:
        :param target_limit:
        :param num_epoches:
        :param batch_size:
        :return:
        '''
        ids = range(len(data))
        context_idxs = []
        seg_indexs = []
        target_indexs = []
        targets = []
        labels = []
        for epoch in range(num_epoches):
            permutation = np.random.permutation(ids)
            for i, id in enumerate(permutation):
                up_context = data[id]['up_source']
                down_context = data[id]['down_source']
                target = data[id]['target']
                cit_target = data[id]['cit_target']

                if len(target) == 0:
                    continue

                if random.randint(0,9) >= 5:
                    context_idx, target_index, target , seg_index= self.process(up_context, down_context, target,
                                                                              context_limit)
                    labels.append(1)
                else:
                    context_idx, target_index, target , seg_index= self.process(up_context, down_context, cit_target,
                                                                              context_limit)
                    labels.append(0)
                context_idxs.append(context_idx)
                target_indexs.append(target_index)
                targets.append(target)
                seg_indexs.append(seg_index)
                if len(context_idxs) == batch_size or ((i == (len(permutation) - 1) and epoch == (num_epoches - 1))):
                    batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'seg_indexs': torch.LongTensor(np.array(seg_indexs, dtype=np.int64)),
                        'target_indexs': target_indexs ,
                        'targets': torch.LongTensor(np.array(targets, dtype=np.int64)),
                        'labels': torch.LongTensor(np.array(labels, dtype=np.int64))
                    }
                    context_idxs = []
                    target_indexs = []
                    seg_indexs = []
                    targets = []
                    labels = []
                    yield batch, epoch

    def double_train_batch(self, data, context_limit, num_epoches, batch_size):
        '''
        [
            {
             "up_source":"",
             "down_source":"",
             "target":""
             "high_dic":{
                        "cit_up_source":"",
                        "cit_down_source":"",
                        "cit_target":"",
                        "bleu1_score":""
                    },
             "low_dic":{
                        "cit_up_source":"",
                        "cit_down_source":"",
                        "cit_target":"",
                        "bleu1_score":""
                    }
            }
        ]
        :param data:
        :param context_limit:
        :param target_limit:
        :param num_epoches:
        :param batch_size:
        :return:
        '''
        ids = range(len(data))
        high_cit_context_idxs = []
        high_source_context_idxs = []
        low_cit_context_idxs = []
        low_source_context_idxs = []
        high_seg_indexs = []
        low_seg_indexs = []
        high_source_seg_indexs = []
        low_source_seg_indexs = []
        for epoch in range(num_epoches):
            permutation = np.random.permutation(ids)
            for i, id in enumerate(permutation):
                up_context = data[id]['up_source']
                down_context = data[id]['down_source']
                target = data[id]['target']
                high_dic = data[id]['high_dic']
                low_dic = data[id]['low_dic']

                if len(target) == 0:
                    continue

                cit_up_context = high_dic['cit_up_source']
                cit_down_context = high_dic['cit_down_source']
                cit_target = high_dic['cit_target']

                context_idx, seg_indx = self.train_process(up_context, "", "", context_limit)
                high_cit_context_idxs.append(context_idx)
                high_seg_indexs.append(seg_indx)

                context_idx, seg_indx = self.train_process(cit_up_context, cit_down_context, cit_target, context_limit)
                high_source_context_idxs.append(context_idx)
                high_source_seg_indexs.append(seg_indx)

                cit_up_context = low_dic['cit_up_source']
                cit_down_context = low_dic['cit_down_source']
                cit_target = low_dic['cit_target']

                context_idx,seg_indx = self.train_process(up_context, "", "", context_limit)
                low_cit_context_idxs.append(context_idx)
                low_seg_indexs.append(seg_indx)

                context_idx,seg_indx = self.train_process(cit_up_context, cit_down_context, cit_target, context_limit)
                low_source_context_idxs.append(context_idx)
                low_source_seg_indexs.append(seg_indx)

                if len(high_cit_context_idxs) == batch_size or ((i == (len(permutation) - 1) and epoch == (num_epoches - 1))):
                    batch = {
                        'high_cit_context_idxs': torch.LongTensor(np.array(high_cit_context_idxs, dtype=np.int64)),
                        'high_seg_indexs': torch.LongTensor(np.array(high_seg_indexs, dtype=np.int64)),
                        'low_cit_context_idxs': torch.LongTensor(np.array(low_cit_context_idxs, dtype=np.int64)),
                        'low_seg_indexs': torch.LongTensor(np.array(low_seg_indexs, dtype=np.int64)),
                        'high_source_context_idxs': torch.LongTensor(np.array(high_source_context_idxs, dtype=np.int64)),
                        'high_source_seg_indexs': torch.LongTensor(np.array(high_source_seg_indexs, dtype=np.int64)),
                        'low_source_context_idxs': torch.LongTensor(np.array(low_source_context_idxs, dtype=np.int64)),
                        'low_source_seg_indexs': torch.LongTensor(np.array(low_source_seg_indexs, dtype=np.int64))
                    }
                    high_cit_context_idxs = []
                    high_seg_indexs = []
                    low_cit_context_idxs = []
                    low_seg_indexs = []
                    high_source_context_idxs = []
                    high_source_seg_indexs = []
                    low_source_context_idxs = []
                    low_source_seg_indexs = []
                    yield batch, epoch

    def dev_batch(self, data, context_limit):
        '''
        {
             "citStr":"",
             "context":"",
             "up_source":"",
             "down_source":"",
             "target":""
             "citations":[
                          citation0,
                          citation1,
                           ...
                          ]
            }
        :param data:
        :param context_limit:
        :param target_limit:
        :param num_epoches:
        :param batch_size:
        :return:
        '''
        ids = range(len(data))
        context_idxs = []
        seg_indexs = []
        source_context_idxs = []
        source_seg_indexs = []
        cit_targets =[]
        ref_labels = []
        for id in ids:
            # up_context = data[id]['up_source_tokens']
            up_context = "Other work attempts to alleviate the data requirements for semantic role labeling either by relying on unsupervised learning or by extending existing resources through the use of unlabeled data ."
            down_context = data[id]['down_source_tokens']
            target = data[id]['target_tokens']
            citations = data[id]['citations_tokens']
            # new_citations = getTopSimilar(data[id], self.idf_dic, top =3)
            for i in range(len(citations)):
                citation = citations[i]
                if citation['label'] == 1:
                    ref_labels.append(i)
            for i in range(len(citations)):#
                # if i > 15:
                #     break
                citation = citations[i]
                # cit_up_context = citation['up_source_tokens']
                cit_up_context = "Beyond annotation projection, Gordon and Swanson ( 2007 ) propose to increase the coverage of PropBank to unseen verbs by finding syntactically similar ( labeled ) verbs and using their annotations as surrogate training data ."
                # cit_down_context = citation['down_source_tokens']
                cit_down_context = "Their algorithm induces role labels following a bootstrapping scheme where the set of labeled instances is iteratively expanded using a classifier trained on previously labeled instances ."
                # cit_target = citation['target_tokens']
                cit_target = "Swier and Stevenson ( 2004 ) present an unsupervised method for labeling the arguments of verbs with their semantic roles ."
                context_idx,  seg_indx = self.train_process(up_context, "", cit_target, context_limit)
                context_idxs.append(context_idx)
                seg_indexs.append(seg_indx)
                context_idx,  seg_indx = self.train_process(cit_up_context, cit_down_context, cit_target, context_limit)
                source_context_idxs.append(context_idx)
                source_seg_indexs.append(seg_indx)
                cit_targets.append(cit_target)

                batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'seg_indexs': torch.LongTensor(np.array(seg_indexs, dtype=np.int64)),
                        'source_context_idxs': torch.LongTensor(np.array(source_context_idxs, dtype=np.int64)),
                        'source_seg_indexs': torch.LongTensor(np.array(source_seg_indexs, dtype=np.int64)),
                        'cit_targets':cit_targets,
                        'targets':target,
                        'ref_labels':ref_labels,
                        'id':id
                    }
                context_idxs = []
                cit_targets = []
                seg_indexs = []
                source_context_idxs = []
                source_seg_indexs =[]
                ref_labels = []
                yield batch

    def lm_dev_batch(self, data, context_limit):
        '''
        {
             "citStr":"",
             "context":"",
             "up_source":"",
             "down_source":"",
             "target":""
             "citations":[
                          citation0,
                          citation1,
                           ...
                          ]
            }
        :param data:
        :param context_limit:
        :param target_limit:
        :param num_epoches:
        :param batch_size:
        :return:
        '''
        ids = range(len(data))
        context_idxs = []
        seg_indexs = []
        source_context_idxs = []
        source_seg_indexs = []
        cit_targets =[]
        ref_labels = []
        target_indexs = []
        targets = []
        for id in ids:
            up_context = data[id]['up_source_tokens']
            down_context = data[id]['down_source_tokens']
            target = data[id]['target_tokens']
            citations = data[id]['citations_tokens']
            # new_citations = getTopSimilar(data[id], self.idf_dic, top =3)
            for i in range(len(citations)):
                citation = citations[i]
                if citation['label'] == 1:
                    ref_labels.append(i)
            for i in range(1):#len(citations)
                # if i > 15:
                #     break
                citation = citations[i]
                cit_up_context = citation['up_source_tokens']
                cit_down_context = citation['down_source_tokens']
                cit_target = citation['target_tokens']
                context_idx, target_index, target, seg_indx = self.process(up_context, down_context, target, context_limit)
                target_indexs.append(target_index)
                targets.append(target)
                context_idxs.append(context_idx)
                seg_indexs.append(seg_indx)
                context_idx,  seg_indx = self.train_process(cit_up_context, cit_down_context, cit_target, context_limit)
                source_context_idxs.append(context_idx)
                source_seg_indexs.append(seg_indx)
                cit_targets.append(cit_target)

                batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'seg_indexs': torch.LongTensor(np.array(seg_indexs, dtype=np.int64)),
                        'source_context_idxs': torch.LongTensor(np.array(source_context_idxs, dtype=np.int64)),
                        'source_seg_indexs': torch.LongTensor(np.array(source_seg_indexs, dtype=np.int64)),
                        'cit_targets':cit_targets,
                        'target_indexs':target_indexs,
                        'targets':torch.LongTensor(np.array(targets, dtype=np.int64)),
                        'ref_labels':ref_labels,
                        'id':id
                    }
                context_idxs = []
                cit_targets = []
                seg_indexs = []
                source_context_idxs = []
                source_seg_indexs =[]
                ref_labels = []
                targets = []
                target_indexs = []
                yield batch

    def citation_double_train_batch(self, data, context_limit, citation_limit, num_epoches, batch_size):
        '''
                [
                    {
                     "up_source":"",
                     "down_source":"",
                     "target":""
                     "high_dic":{
                                "cit_up_source":"",
                                "cit_down_source":"",
                                "cit_target":"",
                                "bleu1_score":""
                            },
                     "low_dic":{
                                "cit_up_source":"",
                                "cit_down_source":"",
                                "cit_target":"",
                                "bleu1_score":""
                            }
                    }
                ]
                :param data:
                :param context_limit:
                :param target_limit:
                :param num_epoches:
                :param batch_size:
                :return:
                '''
        ids = range(len(data))
        context_idxs = []
        high_cit_context_idxs = []
        low_cit_context_idxs = []
        targets = []
        high_citation_idxs = []
        low_citation_idxs = []
        scores = []
        for epoch in range(num_epoches):
            permutation = np.random.permutation(ids)
            for i, id in enumerate(permutation):
                up_context = data[id]['up_source']
                down_context = data[id]['down_source']
                target = data[id]['target']
                high_dic = data[id]['high_dic']
                low_dic = data[id]['low_dic']

                if len(target) == 0:
                    continue

                cit_up_context = high_dic['cit_up_source']
                cit_down_context = high_dic['cit_down_source']
                cit_target = high_dic['cit_target']

                context_idx, high_cit_context_idx, high_citation_idx = self.process(up_context, down_context,
                                                                                    cit_up_context, cit_down_context,
                                                                                    cit_target, context_limit,
                                                                                    citation_limit)
                high_cit_context_idxs.append(high_cit_context_idx)
                high_citation_idxs.append(high_citation_idx)

                cit_up_context = low_dic['cit_up_source']
                cit_down_context = low_dic['cit_down_source']
                cit_target = low_dic['cit_target']

                _, low_cit_context_idx, low_citation_idx = self.process(up_context, down_context, cit_up_context,
                                                                        cit_down_context, cit_target, context_limit,
                                                                        citation_limit)

                context_idxs.append(context_idx)
                low_cit_context_idxs.append(low_cit_context_idx)
                low_citation_idxs.append(low_citation_idx)
                targets.append(target)


                if len(context_idxs) == batch_size or ((i == (len(permutation) - 1) and epoch == (num_epoches - 1))):
                    batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'high_cit_context_idxs': torch.LongTensor(np.array(high_cit_context_idxs, dtype=np.int64)),
                        'high_citation_idxs': torch.LongTensor(np.array(high_citation_idxs, dtype=np.int64)),
                        'low_cit_context_idxs': torch.LongTensor(np.array(low_cit_context_idxs, dtype=np.int64)),
                        'low_citation_idxs': torch.LongTensor(np.array(low_citation_idxs, dtype=np.int64)),
                        'scores': torch.Tensor(np.array(scores, dtype=np.float)),
                        'targets': targets
                    }
                    context_idxs = []
                    high_cit_context_idxs = []
                    high_citation_idxs = []
                    low_cit_context_idxs = []
                    low_citation_idxs = []
                    targets = []
                    scores = []
                    yield batch, epoch

if __name__ == '__main__':

    pass


