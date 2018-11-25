#!usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on 2018年10月23日
@author: sui
'''
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from process import picklesave


def position_encoding_init(n_position, d_pos_vec):
    '''
    Init the sinusoid position encoding table
    '''
    #n_position:句子长度
    #d_pos_vec:维度
    # keep dim 0 for padding token position encoding zero vector
    #//取结果最小的整数
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.cuda.FloatTensor)

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
            self.embedding.weight = nn.Parameter(pretrained, requires_grad=True)

    #         self.embedding.weight.data.uniform_(-initrange,initrange)

    def getEmbedding(self, input):
        return self.embedding(input)

'''
进行self-attention操作
'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,attn_dropout = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper  #64
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)#(n_head*mb_size) x len_q x len_k
        attn = self.dropout(attn)
        #v : (n_head*mb_size) x len_v x d_v
        #att : (n_head*mb_size) x len_q x d_k
        output = torch.bmm(attn, v)#(n_head*mb_size) x len_v x d_k
        return output, attn

'''
层级别的正则化
layerNorm
'''
class LayerNormalization(nn.Module):
    ''' Layer normalization module '''
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        #d_hid 512
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        #z : outputs + residual
        #z.size():mb_size * len_v * d_k
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out

'''
MultiHeadAttention
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        '''

        :param q: batch * length * d_model
        :param k:
        :param v:
        :param attn_mask:
        :return:
        '''
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s)
        #outputs:(n_head*mb_size) x 1en_v x d_k

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)   #mb_size * len_v * (d_k* n_head)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        #outputs:mb_size * len_v * d_model
        result = self.layer_norm(outputs + residual)
        return result, attns

'''
前向计算的神经网络
'''
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x:batch * len * dim
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

'''
encoder端的 一个encoder计算
'''
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            args.n_head, args.d_model, args.d_k, args.d_v, dropout=args.dropout)
        self.pos_ffn = PositionwiseFeedForward(args.d_model, args.d_inner_hid, dropout=args.dropout)  # 512   1024

    def forward(self, enc_input, slf_attn_mask=None):
        '''

        :param enc_input: batch * len * d_model
        :param slf_attn_mask:
        :return: batch * len * d_model
        '''
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

'''
decoder端的一个decoder
'''
class DecoderLayer(nn.Module):
    def __init__(self,args):
        super(DecoderLayer,self).__init__()
        self.slf_attn = MultiHeadAttention(args.n_head, args.d_model, args.d_k, args.d_v, dropout=args.dropout)
        self.enc_attn = MultiHeadAttention(args.n_head, args.d_model, args.d_k, args.d_v, dropout=args.dropout)
        self.pos_ffn = PositionwiseFeedForward(args.d_model, args.d_inner_hid, dropout=args.dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):  #q k v
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

'''
编码端
'''
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, src_seq, return_attns=False):
        enc_slf_attns = []
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output

'''
解码端
'''
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self,dec_input, enc_output,return_attns = False):
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
        return dec_output

class Transformer(nn.Module):
    '''
    A sequence to sequence model with attention mechanism.
    parser.add_argument('--d_model', default=512)
    parser.add_argument('--d_inner_hid', default=1024)
    parser.add_argument('--n_layers', default=6)
    parser.add_argument('--n_head', default=8)
    parser.add_argument('--d_k', default=60)
    parser.add_argument('--d_v', default=60)
    '''
    def __init__(self, args, word_vec):
        super(Transformer, self).__init__()
        self.Tokenembedding = nn.Embedding(word_vec, args.d_model, padding_idx=0)
        self.Segembedding = nn.Embedding(4, args.d_model, padding_idx=0)
        self.position_enc = nn.Embedding(args.length, args.d_model, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(args.length, args.d_model)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.tgt_word_proj = nn.Linear(args.d_model, word_vec, bias=True)
        self.sent_proj = nn.Linear(args.d_model, 2, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def getTransforEmbedding(self, sentence, sentence_Seg, sentence_mask):
        content_embedding = self.Tokenembedding(sentence) + self.Segembedding(sentence_Seg)
        s_lens = torch.sum(sentence_mask.type(torch.cuda.LongTensor), dim=1).data.cpu().numpy()
        sentence_pos = torch.zeros(sentence_mask.size(0), sentence_mask.size(1)).cuda().type(torch.cuda.LongTensor)
        for i in range(len(s_lens)):
            for j in range(s_lens[i]):
                sentence_pos[i][j] = j
        content_embedding += self.position_enc(sentence_pos)
        # enc_output:batch * len * d_model
        enc_output = self.encoder(content_embedding)
        return enc_output

    def predict(self, sentence, sentence_Seg, sentence_mask):
        content_embedding = self.Tokenembedding(sentence) + self.Segembedding(sentence_Seg)
        s_lens = torch.sum(sentence_mask.type(torch.cuda.LongTensor), dim=1).data.cpu().numpy()
        sentence_pos = torch.zeros(sentence_mask.size(0), sentence_mask.size(1)).cuda().type(torch.cuda.LongTensor)
        for i in range(len(s_lens)):
            for j in range(s_lens[i]):
                sentence_pos[i][j] = j
        content_embedding += self.position_enc(sentence_pos)
        # enc_output:batch * len * d_model
        enc_output = self.encoder(content_embedding)
        sent_logit = self.sent_proj(enc_output[:, 0, :])
        return F.log_softmax(sent_logit, dim=1)

    def forward(self, sentence, sentence_Seg, sentence_mask, target_indexs):
        '''
        :param sentence: batch * index
        :param sentence_mask: batch * len
        :param target_indexs: index的集合[] ， 选择预测的目标词
        :return:
        '''
        content_embedding = self.Tokenembedding(sentence) + self.Segembedding(sentence_Seg)
        s_lens = torch.sum(sentence_mask.type(torch.cuda.LongTensor), dim=1).data.cpu().numpy()
        sentence_pos = torch.zeros(sentence_mask.size(0), sentence_mask.size(1)).cuda().type(torch.cuda.LongTensor)
        for i in range(len(s_lens)):
            for j in range(s_lens[i]):
                sentence_pos[i][j] = j
        content_embedding += self.position_enc(sentence_pos)
        #enc_output:batch * len * d_model
        enc_output = self.encoder(content_embedding)

        #选择预测的目标词
        target_words = []
        # print(target_indexs)
        for i in range(len(target_indexs)):
            target_word = []
            for j in range(len(target_indexs[i])):
                target_word.append(enc_output[i].select(dim=0, index=target_indexs[i][j]).unsqueeze(0))
            target_words.append(torch.cat(target_word, dim=0).unsqueeze(0))
        dec_output = torch.cat(target_words, dim=0)
        seq_logit = self.tgt_word_proj(dec_output)
        sent_logit = self.sent_proj(enc_output[:,0,:])
        return F.log_softmax(seq_logit, dim=2), F.log_softmax(sent_logit, dim=1)

class Classify(nn.Module):
    def __init__(self, args, transform):
        super().__init__()
        self.transform = transform
        self.linear = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, sentence, sentence_Seg, sentence_mask):
        inputs = self.transform.getTransforEmbedding(sentence, sentence_Seg, sentence_mask)
        return self.linear( inputs[:,0,:] )

class AllClassify(nn.Module):
    def __init__(self, args, transform):
        super().__init__()
        self.transform = transform
        self.linear = nn.Sequential(
            nn.Linear(2 * args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, sentence, sentence_Seg, sentence_mask, source_sentence, source_sentence_Seg, source_sentence_mask):
        inputs1 = self.transform.getTransforEmbedding(sentence, sentence_Seg, sentence_mask)
        inputs2 = self.transform.getTransforEmbedding(source_sentence, source_sentence_Seg, source_sentence_mask)
        inputs1 = inputs1[:,0,:]#torch.mean(inputs1, dim=1)
        inputs2 = inputs2[:,0,:]#torch.mean(inputs2, dim=1)
        # inputs1 = inputs1.unsqueeze(1)
        # inputs2 = inputs2.unsqueeze(1)
        # result = torch.bmm(inputs1, inputs2.transpose(1,2)).squeeze(2)
        # return F.sigmoid(result)
        inputs = torch.cat([inputs1, inputs2], dim=1)
        return self.linear( inputs )

INF = 1e30
def softmax_mask(val, mask):
    return -INF * (1 - mask) + val

class AllClassifyPyramid(nn.Module):
    def __init__(self, args, transform):
        super().__init__()
        self.transform = transform

        # self-attention 参数
        self.W = nn.Linear( args.d_model, args.d_model)
        self.P = nn.Linear(args.d_model, args.d_model)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.uniform_(self.W.bias)
        torch.nn.init.xavier_uniform_(self.P.weight)
        torch.nn.init.uniform_(self.P.bias)
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Sequential(
            nn.Linear( 4*args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 1, bias=True),
            nn.Sigmoid()
        )

    def self_attention(self, content, hidden, content_mask):
        hidden_ = self.W(hidden.unsqueeze(1))
        content_ = self.P(content)
        alphas = torch.bmm(content_, hidden_.transpose(1, 2))
        print("content_mask",content_mask)
        print("content_:",content)
        alphas = softmax_mask(alphas, content_mask.unsqueeze(2).expand_as(alphas))
        print(alphas)
        alphas = F.softmax(alphas, dim=1)
        result = torch.bmm(alphas.transpose(1, 2), content)
        return result.squeeze(1), alphas

    def forward(self, sentence, sentence_Seg, sentence_mask, source_sentence, source_sentence_Seg, source_sentence_mask):
        inputs1 = self.transform.getTransforEmbedding(sentence, sentence_Seg, sentence_mask)
        inputs2 = self.transform.getTransforEmbedding(source_sentence, source_sentence_Seg, source_sentence_mask)

        #方案一：
        # q_lens = torch.sum(sentence_mask, dim=1).type(torch.cuda.LongTensor)
        # q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        # inputs1 = inputs1[:, 0:q_len_max, :]
        # sentence_mask = sentence_mask[:, 0:q_len_max]
        #
        # d_lens = torch.sum(source_sentence_mask, dim=1).type(torch.cuda.LongTensor)
        # d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        # inputs2 = inputs2[:, 0:d_len_max, :]
        # source_sentence_mask = source_sentence_mask[:, 0:d_len_max]
        # sen_1 = inputs1[:, 0, :]
        # sen_2 = inputs1[:, 0, :]
        # sen_11 = self.self_attention(inputs1[:,1:,:], sen_1, sentence_mask[:,1:])
        # sen_22 = self.self_attention(inputs2[:,1:,:], sen_2, source_sentence_mask[:,1:])
        # sen_11 = torch.mean(inputs1[:,1:,:], dim=1)#inputs1[:,1:,:]#
        # sen_22 = torch.mean(inputs2[:,1:,:], dim=1)#inputs2[:,1:,:]#
        # convout = torch.cat([sen_1, sen_11, sen_2, sen_22], dim=1)
        # return self.linear( convout )

        #方案二:
        # sen_1 = inputs1[:,0,:]
        # sen_2 = inputs2[:,0,:]
        # sen_dot = sen_1 * sen_2
        # sen_error = sen_1 - sen_2
        # out = torch.cat([sen_1, sen_2, sen_dot, sen_error], dim=1)
        # return self.linear( out )

        # 方案三：
        q_lens = torch.sum(sentence_mask, dim=1).type(torch.cuda.LongTensor)
        q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        inputs1 = inputs1[:, 0:q_len_max, :]
        sentence_mask = sentence_mask[:, 0:q_len_max]

        d_lens = torch.sum(source_sentence_mask, dim=1).type(torch.cuda.LongTensor)
        d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        inputs2 = inputs2[:, 0:d_len_max, :]
        source_sentence_mask = source_sentence_mask[:, 0:d_len_max]
        inputs1 = self.dropout(inputs1)
        inputs2 = self.dropout(inputs2)

        sen_1 = inputs1[:, 0, :]
        sen_2 = inputs1[:, 0, :]
        sen_11, alpha1 = self.self_attention(inputs2[:,1:,:], sen_1, source_sentence_mask[:,1:])
        sen_22, alpha2 = self.self_attention(inputs1[:,1:,:], sen_2, sentence_mask[:,1:])
        dic = {}
        alpha1 = alpha1.squeeze(1).cpu().data.numpy()
        alpha2 = alpha2.cpu().data.numpy()
        dic["alpha1"] = alpha1
        dic["alpha2"] = alpha2
        print(dic)
        picklesave("../alpha.pkl", dic, " alpha")
        convout = torch.cat([sen_1, sen_11, sen_2, sen_22], dim=1)
        return self.linear( convout )


class AllClassifyGetKeyWords(nn.Module):
    def __init__(self, args, transform):
        super().__init__()
        self.transform = transform
        self.linear = nn.Sequential(
            nn.Linear(2 * args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 1, bias=True),
            nn.Sigmoid()
        )
        self.linearSelect = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, sentence, sentence_Seg, sentence_mask, source_sentence, source_sentence_Seg, source_sentence_mask):
        inputs1 = self.transform.getTransforEmbedding(sentence, sentence_Seg, sentence_mask)
        inputs2 = self.transform.getTransforEmbedding(source_sentence, source_sentence_Seg, source_sentence_mask)
        selectOut1 = self.linearSelect(inputs1[:,1:,:])
        selectOut2 = self.linearSelect(inputs2[:,1:,:])
        inputs1 = inputs1[:,0,:]#torch.mean(inputs1, dim=1)
        inputs2 = inputs2[:,0,:]#torch.mean(inputs2, dim=1)
        # inputs1 = inputs1.unsqueeze(1)
        # inputs2 = inputs2.unsqueeze(1)
        # result = torch.bmm(inputs1, inputs2.transpose(1,2)).squeeze(2)
        # return F.sigmoid(result)
        inputs = torch.cat([inputs1, inputs2], dim=1)
        return self.linear( inputs ), torch.sum(selectOut1, dim=1), torch.sum(selectOut2, dim=1)





