"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
import numpy as np

from Retrieval.models.Lstm_Match import softmax_mask


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add
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
import torch.nn.functional as F
class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self, args, vocab_size, pretrained):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden
        self.dropout = args.dropout

        self.embedding = Embedding(args.embedding_dim, vocab_size, pretrained)

        self._rnn_dropout = nn.Dropout(self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding_q = LSTM(args.embedding_dim, args.hidden, True, num_layers=1, bidirectional=True, dropout=args.dropout)
        self._encoding_d = LSTM(args.embedding_dim, args.hidden, True, num_layers=1, bidirectional=True, dropout=args.dropout)

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition_q = LSTM(self.hidden_size, args.hidden, True, num_layers=1, bidirectional=True, dropout=args.dropout)
        self._composition_d = LSTM(self.hidden_size, args.hidden, True, num_layers=1, bidirectional=True, dropout=args.dropout)

        self._classification = nn.Sequential(
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_size,
                                                       1),
                                             nn.Sigmoid())

        # Initialize all weights and biases in the model.
        # self.apply(_init_esim_weights)

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
        pre_inputs = content
        hyp_inputs = cit_content
        pre_att = content_repre
        hyp_att = cit_content_repre
        error_pre = pre_inputs - pre_att
        error_hyp = hyp_inputs - hyp_att
        mul_pre = torch.mul(pre_inputs, pre_att)
        mul_hyp = torch.mul(hyp_inputs, hyp_att)
        m_pre = torch.cat((pre_inputs, pre_att, error_pre, mul_pre), dim=2)  # , error_pre, mul_pre
        m_hyp = torch.cat((hyp_inputs, hyp_att, error_hyp, mul_hyp), dim=2)  #
        return m_pre, m_hyp, match

    def forward(self, premises, hypotheses, content_mask, cit_content_mask):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        embedded_premises = self.embedding.getEmbedding(premises)
        embedded_hypotheses = self.embedding.getEmbedding(hypotheses)


        embedded_premises = self._rnn_dropout(embedded_premises)
        embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        q_lens = torch.sum(content_mask, dim=1).type(torch.cuda.LongTensor)
        q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        embedded_premises = embedded_premises[:, 0:q_len_max, :]
        content_mask = content_mask[:, 0:q_len_max]

        d_lens = torch.sum(cit_content_mask, dim=1).type(torch.cuda.LongTensor)
        d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        embedded_hypotheses = embedded_hypotheses[:, 0:d_len_max, :]
        cit_content_mask = cit_content_mask[:, 0:d_len_max]

        encoded_premises,_ = self._encoding_q(embedded_premises,
                                          q_lens)
        encoded_hypotheses,_ = self._encoding_d(embedded_hypotheses,
                                            d_lens)

        enhanced_premises, enhanced_hypotheses , _=\
            self.attention(encoded_premises,
                            encoded_hypotheses, content_mask, cit_content_mask)

        enhanced_premises = self._projection(enhanced_premises)
        enhanced_hypotheses = self._projection(enhanced_hypotheses)

        v_ai,_ = self._composition_q(enhanced_premises, q_lens)
        v_bj,_ = self._composition_d(enhanced_hypotheses, d_lens)

        v_a_avg = torch.mean(v_ai, dim=1)
        v_b_avg = torch.mean(v_bj, dim=1)

        v_a_max = torch.max(v_ai, dim=1)[0]
        v_b_max = torch.max(v_bj, dim=1)[0]

        v = torch.cat([v_a_avg,  v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)

        return logits


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
