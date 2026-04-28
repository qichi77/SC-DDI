import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions
    

class RESCAL(nn.Module):

    def __init__(self, n_features, depth):
        super().__init__()
        self.n_features = n_features
        self.co_attn = CoAttentionLayer(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(depth*depth, 2)
        )

    def forward(self, heads, tails):
        alpha_scores = self.co_attn(heads, tails)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        scores = (heads @ tails.transpose(-2, -1))
        scores *= alpha_scores
        scores = self.mlp(scores.reshape(scores.shape[0], -1))
        return scores
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"
    
class PoolAttention(nn.Module):
    """利用Attention进行多模态融合, `with-attn`变体的关键组件"""

    def __init__(self, n_features, num_neads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_features, num_heads=num_neads, batch_first=True)
        self.drug_norm = nn.LayerNorm(n_features)
        self.prot_norm = nn.LayerNorm(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features*2, n_features*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*2, n_features*1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*1, 2),
        )

    def forward(self, drug, prot):
        drug = self.drug_norm(drug)
        prot = self.prot_norm(prot)
        drug_attn = self.attn(drug, prot, prot)[0]
        prot_attn = self.attn(prot, drug, drug)[0]
        drug_pool = torch.max((drug+drug_attn)/2, dim=1)[0]
        prot_pool = torch.max((prot+prot_attn)/2, dim=1)[0]
        scores = self.mlp(torch.cat([drug_pool, prot_pool], dim=-1))
        return scores

class AttentionLayer(nn.Module):
    def __init__(self, n_features, heads=4):
        super().__init__()
        self.n_features = n_features
        self.heads = heads
        self.attn = nn.MultiheadAttention(self.n_features, self.heads, batch_first=True, dropout=0.3)
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.n_features*2, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, 2)
        )

    def forward(self, drug_repr, prot_repr):
        drug_output, _ = self.attn(drug_repr, prot_repr, prot_repr)
        prot_output, _ = self.attn(prot_repr, drug_repr, drug_repr)

        drug_output = drug_output * 0.5 + drug_repr * 0.5
        prot_output = prot_output * 0.5 + prot_repr * 0.5

        drug_pool, _ = torch.max(drug_output, dim=1)
        prot_pool, _ = torch.max(prot_output, dim=1)
        concat_repr = torch.cat([drug_pool, prot_pool], -1)
        result = self.mlp(concat_repr)
        return result
    

def rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_CNNs(input_dim, conv_dim, kernel):
    return nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=kernel[2]),
            nn.ReLU(),
        )
