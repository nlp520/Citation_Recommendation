#-*- coding:utf-8 -*-
import pickle

from process import pickleload
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords

class Batch():
    def __init__(self, args):
        self.OOV = 0
        self.OOD = 1 #unknow
        self.word2index = pickleload(args.word2index_pkl, "word2index")
        self.stoplis = {}
        for word in stopwords.words("english"):
            self.stoplis[word] = 1
    def getindex(self, word):
        if word in self.word2index:
            return  self.word2index[word]
        else:
            return self.OOD

    def process(self, up_sources, down_sources, cit_up_sources, cit_down_sources, cit_target, context_limit, citation_limit):
        content_source = up_sources.split(" ") + cit_target.split(" ") +down_sources.split(" ")
        cit_content_source = cit_up_sources.split(" ") + cit_target.split(" ") + cit_down_sources.split(" ")
        citation_tokens = cit_target.split(" ")
        # content_source = [token for token in content_source if token not in self.stoplis]
        # cit_content_source = [token for token in cit_content_source if token not in self.stoplis]
        #对
        # tagged_text = nltk.pos_tag(content_source)
        # # content_source = [token[0] for token in tagged_text if "NN" in token[1] or "VB" in token[1]]
        # print(cit_content_source)
        # tagged_text = nltk.pos_tag(" ".join(cit_content_source), )
        # # cit_content_source = [token[0] for token in tagged_text if "NN" in token[1] or "VB" in token[1]]

        while ' 'in citation_tokens:
            citation_tokens.remove(" ")
        while ' 'in content_source:
            content_source.remove(" ")
        while ' 'in cit_content_source:
            cit_content_source.remove(" ")
        context_idxs = np.zeros([context_limit], dtype=np.int64)
        cit_context_idxs = np.zeros([context_limit], dtype=np.int64)
        citation_idxs = np.zeros([citation_limit], dtype=np.int64)

        min_context = len(content_source) if len(content_source) < context_limit else context_limit
        min_cit_context = len(cit_content_source) if len(cit_content_source) < context_limit else context_limit
        min_citation = len(citation_tokens) if len(citation_tokens) < citation_limit else citation_limit

        for i in range(min_context):
            context_idxs[i] = self.getindex(content_source[i])
        for i in range(min_cit_context):
            cit_context_idxs[i] = self.getindex(cit_content_source[i])
        for i in range(min_citation):
            citation_idxs[i] = self.getindex(citation_tokens[i])

        return context_idxs, cit_context_idxs, citation_idxs

    def train_batch(self, data, context_limit, citation_limit, num_epoches, batch_size):
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
        cit_context_idxs = []
        targets = []
        citation_idxs = []
        scores = []
        for epoch in range(num_epoches):
            permutation = np.random.permutation(ids)
            for i, id in enumerate(permutation):
                up_context = data[id]['up_source']
                down_context = data[id]['down_source']
                target = data[id]['target']
                cit_up_context = data[id]['cit_up_source']
                cit_down_context = data[id]['cit_down_source']
                cit_target = data[id]['cit_target']
                score = data[id]['bleu1_score']

                if len(target) ==0:
                    continue
                context_idx, cit_context_idx, citation_idx = self.process(up_context, down_context,cit_up_context, cit_down_context, cit_target, context_limit, citation_limit)

                context_idxs.append(context_idx)
                cit_context_idxs.append(cit_context_idx)
                citation_idxs.append(citation_idx)
                targets.append(target)
                scores.append([score])
                if len(context_idxs) == batch_size or ((i == (len(permutation) - 1) and epoch == (num_epoches - 1))):
                    batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'cit_context_idxs': torch.LongTensor(np.array(cit_context_idxs, dtype=np.int64)),
                        'citation_idxs': torch.LongTensor(np.array(citation_idxs, dtype=np.int64)),
                        'scores': torch.Tensor(np.array(scores, dtype=np.float)),
                        'targets': targets
                    }
                    context_idxs = []
                    cit_context_idxs = []
                    targets = []
                    citation_idxs = []
                    scores = []
                    yield batch, epoch

    def double_train_batch(self, data, context_limit, citation_limit, num_epoches, batch_size):
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

                context_idx, high_cit_context_idx, high_citation_idx = self.process(up_context, down_context, cit_up_context, cit_down_context, cit_target, context_limit, citation_limit)
                high_cit_context_idxs.append(high_cit_context_idx)
                high_citation_idxs.append(high_citation_idx)

                cit_up_context = low_dic['cit_up_source']
                cit_down_context = low_dic['cit_down_source']
                cit_target = low_dic['cit_target']

                _, low_cit_context_idx, low_citation_idx = self.process(up_context, down_context, cit_up_context, cit_down_context, cit_target, context_limit, citation_limit)

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

    def dev_batch(self, data, context_limit, citation_limit):
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
        cit_context_idxs = []
        citation_idxs = []
        cit_targets = []
        ref_labels = []
        for id in ids:
            up_context = data[id]['up_source_tokens']
            down_context = data[id]['down_source_tokens']
            target = data[id]['target_tokens']
            citations = data[id]['citations_tokens']
            for i in range(len(citations)):
                citation = citations[i]
                cit_up_context = citation['up_source_tokens']
                cit_down_context = citation['down_source_tokens']
                cit_target = citation['target_tokens']
                context_idx, cit_context_idx, citation_idx = self.process(up_context, down_context, cit_up_context,
                                                                          cit_down_context, cit_target, context_limit,
                                                                          citation_limit)
                if citation['label'] == 1:
                    ref_labels.append(i)
                context_idxs.append(context_idx)
                cit_context_idxs.append(cit_context_idx)
                cit_targets.append(cit_target)
                citation_idxs.append(citation_idx)
            batch = {
                        'context_idxs': torch.LongTensor(np.array(context_idxs, dtype=np.int64)),
                        'cit_context_idxs': torch.LongTensor(np.array(cit_context_idxs, dtype=np.int64)),
                        'targets': target,
                        'cit_targets': cit_targets,
                        'citation_idxs': torch.LongTensor(np.array(citation_idxs, dtype=np.int64)),
                        'ref_labels': ref_labels,
                        'citations':citations
                    }
            context_idxs = []
            cit_context_idxs = []
            citation_idxs = []
            cit_targets = []
            ref_labels = []
            yield batch


if __name__ == '__main__':
    dic = {}
    for word in stopwords.words("english"):
        dic[word] = 1
    print(len(dic))
    pass


