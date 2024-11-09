import torch
import torch.nn as nn
import torch.nn.functional as F
from EGAT_layers import SpGraphAttentionLayer

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        '''
        Function Description:
        Initializes the model by defining the size of the feature space, and sets up layers for encoding decision variables, edge features, and constraint features. 
        It includes two semi-convolutional attention layers and a final output layer.
        - nfeat: Initial feature dimension.
        - nhid: Dimension of the hidden layers.
        - nclass: Number of classes; for 0-1 integer programming, this would be 2.
        - dropout: Dropout rate.
        - alpha: Coefficient for leakyReLU.
        - nheads: Number of heads in the multi-head attention mechanism.
        Hint: Use the pre-written SpGraphAttentionLayer for the attention layers.
        '''
        super(SpGAT, self).__init__()
        self.dropout = dropout
        embed_size = 64
        self.input_module = torch.nn.Sequential(
            torch.nn.Linear(nfeat, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
        )
        self.attentions_u_to_v = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_u_to_v):
            self.add_module('attention_u_to_v_{}'.format(i), attention)
        self.attentions_v_to_u = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_v_to_u):
            self.add_module('attention_v_to_u_{}'.format(i), attention)

        self.out_att_u_to_v = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.out_att_v_to_u = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, nclass, bias=False),
            #torch.nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, edgeA, edgeB, edge_feat):
        '''
        Function Description:
        Executes the forward pass using the provided constraint, edge, and variable features, processing them through an EGAT to produce the output.

        Parameters:
        - x: Features of the variable and constraint nodes.
        - edgeA, edgeB: Information about the edges.
        - edge_feat: Features associated with the edges.

        Return: The result after the forward propagation.
        '''
        #print(x)
        x = self.input_module(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(x)
        new_edge = torch.cat([att(x, edgeA, edge_feat)[1] for att in self.attentions_u_to_v], dim=1)
        x = torch.cat([att(x, edgeA, edge_feat)[0] for att in self.attentions_u_to_v], dim=1)
        x = self.out_att_u_to_v(x, edgeA, edge_feat)
        new_edge = torch.mean(new_edge, dim = 1).reshape(new_edge.size()[0], 1)
        #x = self.softmax(x)
        new_edge_ = torch.cat([att(x, edgeB, new_edge)[1] for att in self.attentions_v_to_u], dim=1)
        x = torch.cat([att(x, edgeB, new_edge)[0] for att in self.attentions_v_to_u], dim=1)
        x = self.out_att_v_to_u(x, edgeB, new_edge)
        new_edge_ = torch.mean(new_edge_, dim = 1).reshape(new_edge_.size()[0], 1)

        x = self.output_module(x)
        x = self.softmax(x)

        return x, new_edge_
