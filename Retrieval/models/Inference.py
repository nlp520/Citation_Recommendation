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
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layer_dim, dropout, bidirectional):
        super(RNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        if self.bidirectional == True:
            self.num = 2
        else:
            self.num = 1
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,self.layer_dim,True,True,0,self.bidirectional)
        
    def init_hidden(self, inputs):
        if torch.cuda.is_available():
            return(Variable(torch.zeros(self.layer_dim* self.num, inputs.size(0), self.hidden_dim)).cuda(),
                   Variable(torch.zeros(self.layer_dim* self.num, inputs.size(0), self.hidden_dim)).cuda())
        else:
            return(Variable(torch.zeros(self.layer_dim* self.num, inputs.size(0), self.hidden_dim)),
                   Variable(torch.zeros(self.layer_dim* self.num, inputs.size(0), self.hidden_dim)))
    
    def forward(self, inputs):
        hidden = self.init_hidden(inputs)
        lstm_out,h = self.lstm(inputs, hidden)
        return lstm_out #batch*length*2hidden_dim
    

class Inference(nn.Module):
    def __init__(self, args, word_size, pretrained):
        super(Inference, self).__init__()
        self.embedding = Embedding(args.embedding_dim, word_size, pretrained)
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim
        self.dropout = args.dropout
        self.bidirectional = args.bidirectional
        self.encoderPre = RNN(self.embedding_dim, self.hidden_dim, self.layer_dim, self.dropout, self.bidirectional) 
        self.encoderHyp = RNN(self.embedding_dim, self.hidden_dim, self.layer_dim, self.dropout, self.bidirectional)
        
        self.decoderPre = RNN(self.hidden_dim*2*4, self.hidden_dim, self.layer_dim, self.dropout, self.bidirectional)
        self.decoderHyp = RNN(self.hidden_dim*2*4, self.hidden_dim, self.layer_dim, self.dropout, self.bidirectional)

        self.FC = nn.Sequential(
            nn.Linear(args.hidden * 8, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, 1),
            nn.Sigmoid()
        )
        self.Dropout = nn.Dropout(self.dropout)
        
        
    def attFunction(self, pre_inputs, hyp_inputs):
        #pre_inputs:batch*length*2hidden_dim  
        #hyp_inputs:batch*length*2hidden_dim
        alphas = torch.bmm(pre_inputs, hyp_inputs.transpose(1,2))#batch * pre_len *hyp_len
        alphas_pre = F.softmax(alphas, dim=1)#premise的权重分布
        alphas_hyp = F.softmax(alphas, dim=2)#hyp的权重分布
        pre_att = torch.bmm(alphas_pre, hyp_inputs)
        hyp_att = torch.bmm(alphas_hyp.transpose(1,2), pre_inputs)
        
        error_pre = pre_inputs - pre_att
        error_hyp = hyp_inputs - hyp_att
        mul_pre = torch.mul(pre_inputs, pre_att)
        mul_hyp = torch.mul(hyp_inputs, hyp_att)
        m_pre = torch.cat((pre_inputs, pre_att, error_pre,mul_pre ), dim=2)#, error_pre, mul_pre
        m_hyp = torch.cat((hyp_inputs, hyp_att,error_hyp ,mul_hyp), dim=2)#
        return m_pre, m_hyp

    def forward(self,inputs_pre, inputs_hyp, x=None, y =None):
        inputs_pre = self.embedding.getEmbedding(inputs_pre)
        inputs_hyp = self.embedding.getEmbedding(inputs_hyp)
        inputs_pre = self.Dropout(inputs_pre)
        inputs_hyp = self.Dropout(inputs_hyp)
        pre_out1 = self.encoderPre.forward(inputs_pre)
        hyp_out1 = self.encoderHyp.forward(inputs_hyp)
        m_pre,m_hyp = self.attFunction(pre_out1, hyp_out1)

        pre_out1 = self.decoderPre.forward(m_pre)
        hyp_out1 = self.decoderHyp.forward(m_hyp)
        pre_ave = torch.mean(pre_out1, dim=1).squeeze()
        pre_max = torch.max(pre_out1, dim=1)[0].squeeze()
        hyp_ave = torch.mean(hyp_out1,dim=1).squeeze()
        hyp_max = torch.max(hyp_out1, dim=1)[0].squeeze()
        
        pre_hyp = torch.cat((pre_ave,pre_max,hyp_ave,hyp_max), dim=1)
        result = self.FC(pre_hyp)
        
        return result
