#!usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on 2018年5月12日
@author: sui
'''
import sys
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, embedding_dim, word_size, pretrained):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_size = word_size
        self.embedding = nn.Embedding(self.word_size, self.embedding_dim)  # 最后一个全为0,self.word_size
        self.embedding_init(pretrained)

    # embedding层的初始化
    def embedding_init(self, pretrained):
        initrange = 0.1
        if pretrained is not None:
            print("Setting pretrained embedding weights")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            self.embedding.weight.data = pretrained

    #         self.embedding.weight.data.uniform_(-initrange,initrange)

    def getEmbedding(self, input):
        return self.embedding(input)

class Decomposable(nn.Module):
    def __init__(self, args ,word_size, pretrained):#,word_size, pretrained
        super(Decomposable, self).__init__()
        self.Embedding = Embedding(args.embedding_dim, word_size, pretrained)
        self.embedding_dim = args.embedding_dim
        self.linearx = nn.Sequential(
                        nn.Linear(self.embedding_dim, self.embedding_dim),
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim, self.embedding_dim)
                        )
        self.lineary = nn.Sequential(
                        nn.Linear(self.embedding_dim, self.embedding_dim),
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim, self.embedding_dim)
                        )
        self.lineargx = nn.Sequential(
                        nn.Linear(self.embedding_dim * 2 , self.embedding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2)
                        )
        self.lineargy = nn.Sequential(
                        nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2)
                        )

        self.classify = nn.Sequential(
                        nn.Linear(self.embedding_dim * 4, self.embedding_dim ),
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim , 1),
                        nn.Sigmoid()
                        )
        self.dropout = nn.Dropout(args.dropout)


    def attFunction(self, pre_inputs, hyp_inputs):
        #pre_inputs:batch*length*hidden_dim
        #hyp_inputs:batch*length*hidden_dim
        alphas = F.relu(torch.bmm(pre_inputs, hyp_inputs.transpose(1,2)))#batch * pre_len *hyp_len
        alphas_pre = F.softmax(alphas, dim=1)#premise的权重分布
        alphas_hyp = F.softmax(alphas, dim=2)#hyp的权重分布
        pre_att = torch.bmm(alphas_pre, hyp_inputs)
        hyp_att = torch.bmm(alphas_hyp.transpose(1,2), pre_inputs)

        m_pre = torch.cat((pre_inputs, pre_att), dim=2)
        m_hyp = torch.cat((hyp_inputs, hyp_att), dim=2)

        return m_pre, m_hyp 

    def forward(self,inputs_pre, inputs_hyp ,content_mask=None, cit_content_mask=None):
        content = self.Embedding.getEmbedding(inputs_pre)
        cit_content = self.Embedding.getEmbedding(inputs_hyp)
        if self.type == "train":
            content = self.dropout(content)
            cit_content = self.dropout(cit_content)

        q_lens = torch.sum(content_mask, dim=1).type(torch.cuda.LongTensor)
        q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        inputs_pre = content[:, 0:q_len_max, :]
        content_mask = content_mask[:, 0:q_len_max]

        d_lens = torch.sum(cit_content_mask, dim=1).type(torch.cuda.LongTensor)
        d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        inputs_hyp = cit_content[:, 0:d_len_max, :]
        cit_content_mask = cit_content_mask[:, 0:d_len_max]
        pre_out1 = inputs_pre#self.linearx(inputs_pre) # batch * length * embedding
        hyp_out1 = inputs_hyp#self.lineary() # batch * length * embedding
        m_pre, m_hyp = self.attFunction(pre_out1, hyp_out1)
        pre_out2 = m_pre
        hyp_out2 = m_hyp
        pre_sum = torch.sum(pre_out2, dim=1).squeeze()
        # pre_max = torch.max(pre_out2, dim=1)[0].squeeze()
        hyp_sum = torch.sum(hyp_out2,dim=1).squeeze()
        # hyp_max = torch.max(hyp_out2, dim=1)[0].squeeze()
        
        pre_hyp = torch.cat((pre_sum,hyp_sum), dim=1)
        result =self.classify(pre_hyp)
        
        return result
