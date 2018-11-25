
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, embedding_dim, word_size, pretrained=None, pretrained_flag = False):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_size = word_size
        self.embedding = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0)  #
        self.embedding_init(pretrained)

    # embedding层的初始化
    def embedding_init(self, pretrained):
        if pretrained is not None:
            print("Setting pretrained embedding weights")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            self.embedding.weight = nn.Parameter(pretrained, requires_grad=True)
            # print(self.embedding.weight.data[0])
    #         self.embedding.weight.data.uniform_(-initrange,initrange)
    def getEmbedding(self, input):
        return self.embedding.forward(input)

class MatchPyramid(nn.Module):
    def __init__(self, args, word_vec, word_embeddings):
        super(MatchPyramid, self).__init__()
        self.word_embeddings = word_embeddings
        self.Embedding = Embedding(args.embedding_dim, word_vec, word_embeddings, True)
        self.dropout = nn.Dropout(args.dropout)
        self.conv2D = nn.Sequential(\
            nn.Conv2d(1,32,kernel_size=(3, 3), padding=(1, 1)),\
            nn.ReLU(),
            nn.MaxPool2d((3,3))
        )
        self.conv2D2 = nn.Sequential(\
            nn.Conv2d(32, 32,kernel_size=(3, 3)),\
            nn.ReLU(),
            nn.MaxPool2d((3,3))
        )
        self.FC = nn.Sequential(
            nn.Linear(8192, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, content, cit_content, content_mask = None, cit_content_mask = None):
        content = self.Embedding.getEmbedding(content)
        cit_content = self.Embedding.getEmbedding(cit_content)

        content = self.dropout(content)
        cit_content = self.dropout(cit_content)

        # q_lens = torch.sum(content_mask, dim=1).type(torch.cuda.LongTensor)
        # q_len_max = int(torch.max(q_lens, dim=0)[0].cpu().data.numpy())
        # content = content[:, 0:q_len_max, :]
        # content_mask = content_mask[:, 0:q_len_max]
        #
        # d_lens = torch.sum(cit_content_mask, dim=1).type(torch.cuda.LongTensor)
        # d_len_max = int(torch.max(d_lens, dim=0)[0].cpu().data.numpy())
        # cit_content = cit_content[:, 0:d_len_max, :]
        # cit_content_mask = cit_content_mask[:, 0:d_len_max]

        cross = torch.bmm(content, cit_content.transpose(1,2)).unsqueeze(1)
        # den = torch.bmm(torch.sqrt(torch.sum(content * content, dim=2)).unsqueeze(2), \
        #                 torch.sqrt(torch.sum(cit_content * cit_content, dim=2)).unsqueeze(1))
        # cross = (cross/den).unsqueeze(1)
        # batcg * 1 * con_length * cit_len -> batch * out_size * length - 2 * 1
        cross = self.dropout(cross)
        match = self.conv2D(cross)
        match = self.dropout(match)
        match = self.conv2D2(match)

        # conv_out = match.view(match.size(0),match.size(1), -1)
        # conv_out = torch.sum(conv_out, dim=2)
        conv_out = match.view(match.size(0), -1)
        # print(conv_out.size())
        result = self.FC(conv_out)

        return result

if __name__ == '__main__':
    x = torch.Tensor([[0.0003,2.3,4.5,5]]).unsqueeze(1)
    y = torch.Tensor([[0.0003,2.3,4.5,5]]).unsqueeze(2)
    z = torch.bmm(x , y)
    den = torch.bmm(torch.sqrt(torch.sum(x * x, dim=2)).unsqueeze(2), \
                    torch.sqrt(torch.sum(y * y, dim=1)).unsqueeze(1))
    print(z)
    print(den)
    print(z/den)
    # print(x * x)
    pass
