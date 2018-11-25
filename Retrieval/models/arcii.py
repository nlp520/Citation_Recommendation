
import numpy as np
import torch
from torch import nn

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
            self.embedding.weight = nn.Parameter(pretrained, requires_grad=False)
    #         self.embedding.weight.data.uniform_(-initrange,initrange)

    def getEmbedding(self, input):
        return self.embedding(input)

class ArcII(nn.Module):
    def __init__(self, args, word_vec, word_embeddings):
        super(ArcII, self).__init__()
        self.Embedding = Embedding(args.embedding_dim, word_vec, word_embeddings, True)
        self.q_conv1 = nn.Conv1d(1, 32, (3, args.embedding_dim) )
        self.d_conv1 = nn.Conv1d(1, 32, (3, args.embedding_dim) )
        self.dropout = nn.Dropout(args.dropout)
        self.type = args.type
        self.first_conv2D = nn.Sequential(\
            nn.Conv2d(1,32,(3,3)),\
            nn.ReLU(),\
            nn.MaxPool2d(3, 3))
        self.second_conv2D = nn.Sequential(\
            nn.Conv2d(32,32,(3,3)),\
            nn.ReLU(),\
            nn.MaxPool2d(3, 3))

        self.FC = nn.Sequential(
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, content, cit_content, content_mask = None, cit_content_mask = None):
        content = self.Embedding.getEmbedding(content)
        cit_content = self.Embedding.getEmbedding(cit_content)
        if self.type == "train":
            content = self.dropout(content)
            cit_content = self.dropout(cit_content)

        content = content.unsqueeze(1)
        cit_content = cit_content.unsqueeze(1)
        # batcg * 1 * length * embedding_dim -> batch * out_size * length - 2 * 1
        conv_content = self.q_conv1(content).squeeze(3)
        conv_cit_content = self.d_conv1(cit_content).squeeze(3)

        #match 匹配
        match = torch.bmm(conv_content, conv_cit_content.transpose(1, 2))

        first_conv2 = self.first_conv2D(match.unsqueeze(1))
        # print(first_conv2.size())

        second_conv2 = self.second_conv2D(first_conv2)
        # print(second_conv2.size())

        conv_out = second_conv2.view(second_conv2.size(0), -1)

        result = self.FC(conv_out)

        return result
