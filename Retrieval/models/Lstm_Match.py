import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from Retrieval.utils import draw_heatmap

INF = 1e30
def softmax_mask(val, mask):
    return -INF * (1 - mask) + val

class Embedding(nn.Module):
    def __init__(self, embedding_dim, word_size, pretrained=None, pretrained_flag = False):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_size = word_size
        self.embedding = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0)  #
        if pretrained_flag == True:
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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}'.format(i)))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}'.format(i)))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), val=0)
            getattr(self.rnn, 'bias_hh_l{}'.format(i)).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}_reverse'.format(i)))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}_reverse'.format(i)))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}_reverse'.format(i)), val=0)
                getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)).chunk(4)[1].fill_(1)

    def forward(self, x, x_len):
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)
        return x, h

class LstmMatch(nn.Module):
    def __init__(self, args, word_vec, word_embeddings):
        super(LstmMatch, self).__init__()
        self.Embedding = Embedding(args.embedding_dim, word_vec, word_embeddings, True)
        self.Rnn_q = LSTM(args.embedding_dim, args.hidden, True, num_layers=2, bidirectional=True, dropout=args.dropout)
        self.Rnn_d = LSTM(args.embedding_dim, args.hidden, True, num_layers=2, bidirectional=True, dropout=args.dropout)

        #self-attention 参数
        self.W = nn.Linear(2*args.hidden, 2*args.hidden)
        self.P = nn.Linear(2*args.hidden, 2*args.hidden)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.uniform_(self.W.bias)
        torch.nn.init.xavier_uniform_(self.P.weight)
        torch.nn.init.uniform_(self.P.bias)

        self.dropout = nn.Dropout(args.dropout)
        self.FC = nn.Sequential(
            nn.Linear(args.hidden * 8, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, 1),
            nn.Sigmoid()
        )

    def attention(self, content, cit_content, content_mask, cit_content_mask):
        '''
        :param content:目标文献的上下文
        :param cit_content:引用的上下文n
        :param content_mask:mask
        :param cit_content_mask: citation mask
        :return:
        '''
        #batch * q_len * 2hidden   X  batch * d_len * 2hidden ->batch * q_len * d_len
        match = torch.bmm(content, cit_content.transpose(1,2))
        softmax_content = F.softmax(softmax_mask(match, cit_content_mask.unsqueeze(1).expand_as(match)), dim = 2)
        content_repre = torch.bmm(softmax_content, cit_content)

        softmax_cit_content = F.softmax(softmax_mask(match, content_mask.unsqueeze(2).expand_as(match)), dim=1)
        cit_content_repre = torch.bmm(softmax_cit_content.transpose(1, 2), content)

        return content_repre, cit_content_repre, match

    def self_attention(self, content, hidden, content_mask):
        hidden = torch.mean(hidden, dim=1).unsqueeze(1)
        content_ = self.W(content)
        hidden_ = self.P(hidden)
        alphas = torch.bmm(content_, hidden_.transpose(1, 2))
        alphas = softmax_mask(alphas, content_mask.unsqueeze(2).expand_as(alphas))
        alphas = F.softmax(alphas, dim=1)
        result = torch.bmm(alphas.transpose(1, 2), content)
        return result.squeeze(1)

    def forward(self, content, cit_content, content_mask, cit_content_mask):
        content = self.Embedding.getEmbedding(content)
        cit_content = self.Embedding.getEmbedding(cit_content)
        if self.type == "train":
            content = self.dropout(content)
            cit_content = self.dropout(cit_content)

        q_lens = torch.sum(content_mask, dim=1).type(torch.cuda.LongTensor)
        q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        content = content[:, 0:q_len_max, :]
        content_mask = content_mask[:, 0:q_len_max]

        d_lens = torch.sum(cit_content_mask, dim=1).type(torch.cuda.LongTensor)
        d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        cit_content = cit_content[:, 0:d_len_max, :]
        cit_content_mask = cit_content_mask[:, 0:d_len_max]

        q_out, _ = self.Rnn_q.forward(content, q_lens)
        d_out, _ = self.Rnn_d.forward(cit_content, d_lens)

        content_repre, cit_content_repre, match = self.attention(q_out, d_out, content_mask, cit_content_mask)
        content_repre = torch.mean(content_repre, dim=1)
        cit_content_repre = torch.mean(cit_content_repre, dim=1)

        q_out = self.self_attention(q_out, q_out, content_mask)
        d_out = self.self_attention(d_out, d_out, cit_content_mask)

        out = torch.cat([q_out, d_out, content_repre, cit_content_repre], dim=1)
        result = self.FC(out)
        return result
