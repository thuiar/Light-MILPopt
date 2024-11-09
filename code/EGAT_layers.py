import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    def __init__(self, node_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.node_features = node_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(node_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features + 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, node, edge, edge_feature):
        dv = 'cuda:2' if node.is_cuda else 'cpu'
        #dv = 'cpu'
        N = node.size()[0]
        edge = edge.t()
        assert not torch.isnan(edge).any()
        #print(input)

        h = torch.mm(node, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        #print(torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1))
        #print(edge_feature)
        edge_h = torch.cat((torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1), edge_feature), dim = 1).t()
        assert not torch.isnan(edge_h).any()
        # edge: (2*D + 1) x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # attention, edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        #edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        #
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        
        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime)
        h_prime = torch.add(h, h_prime)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        #print(h.size(), h_prime.size())

        if self.concat:
            # if this layer is not last layer,
            return [F.elu(h_prime), edge_e.reshape(edge_e.size()[0], 1)]
        else:
            # if this layer is last layer,
            return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.node_features) + ' -> ' + str(self.out_features) + ')'
