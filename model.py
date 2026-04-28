```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv, SAGPooling, LayerNorm, global_add_pool

from config import hyperparameter
from mamba_ssm import Mamba

from chemberta import ChemBERTaEncoder

class SC_conv_block(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        super(SC_conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)
        self.norm = LayerNorm(self.in_channels)
        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.norm(x, batch))
        x = self.conv(x, edge_index, edge_attr)
        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)
        global_graph_emb = global_add_pool(x, x_batch)
        return x, global_graph_emb


class SCBlock(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.2):
        super(SCBlock, self).__init__()
        
        self.dropout_rate = dropout
        self.hidden_channels = out_channels // (num_heads*2)
        
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=dropout)
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=dropout)
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)
        
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        
        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = GATConv(out_channels, out_channels//num_heads, num_heads)

    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch,
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):
        
        atom_x_res = atom_x
        aa_x_res = aa_x

        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        atom_x = F.dropout(atom_x_res + F.elu(atom_x), self.dropout_rate, self.training)

        aa_x = self.prot_pool(aa_x, aa_edge_index, aa_edge_attr)
        aa_x = F.dropout(aa_x_res + F.elu(aa_x), self.dropout_rate, self.training)

        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class HybridMambaEncoder(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.3):
        super(HybridMambaEncoder, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            groups=d_model
        )
        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x_conv = self.conv(x)
        x_conv = self.act(x_conv)
        
        out = x + self.dropout(x_conv)
        
        return out


class SCBlock_1D(nn.Module):
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12], dropout=0.3):
        super(SCBlock_1D, self).__init__()
        
        self.attention_dim = conv * 4 
        target_dim = self.attention_dim 
        
        self.mix_attention_head = 5

        self.drug_proj = nn.Conv1d(input_dim, target_dim, 1)
        self.prot_proj = nn.Conv1d(input_dim, target_dim, 1)

        self.Drug_Hybrid = HybridMambaEncoder(d_model=target_dim, kernel_size=7, dropout=dropout)
        self.Protein_Hybrid = HybridMambaEncoder(d_model=target_dim, kernel_size=7, dropout=dropout)

        self.mix_attention_layer = nn.MultiheadAttention(
            embed_dim=self.attention_dim, 
            num_heads=self.mix_attention_head, 
            batch_first=True, 
            dropout=dropout
        )

    def forward(self, drugembed, proteinembed):
        
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drug_x = self.drug_proj(drugembed)
        prot_x = self.prot_proj(proteinembed)

        drugConv = self.Drug_Hybrid(drug_x)
        proteinConv = self.Protein_Hybrid(prot_x)

        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)

        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool

class AtomResidueInteraction_Optimized(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(AtomResidueInteraction_Optimized, self).__init__()

        self.hidden_dim = hidden_dim

        self.drug_to_prot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.prot_to_drug_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.drug_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.prot_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.drug_norm1 = nn.LayerNorm(hidden_dim)
        self.drug_norm2 = nn.LayerNorm(hidden_dim)
        self.prot_norm1 = nn.LayerNorm(hidden_dim)
        self.prot_norm2 = nn.LayerNorm(hidden_dim)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) 
        )

    def forward(self, atom_x, aa_x, atom_batch, aa_batch, drug_seq_feat, prot_seq_feat, return_attention=False):
        
        atom_dense, atom_mask = to_dense_batch(atom_x, atom_batch)
        aa_dense, aa_mask = to_dense_batch(aa_x, aa_batch)

        atom_padding_mask = ~atom_mask
        aa_padding_mask = ~aa_mask

        drug_ctx, drug_to_prot_weights = self.drug_to_prot_attn(
            query=atom_dense, key=aa_dense, value=aa_dense, key_padding_mask=aa_padding_mask
        )
        drug_ctx = self.drug_norm1(atom_dense + drug_ctx)
        drug_ctx = self.drug_norm2(drug_ctx + self.drug_ffn(drug_ctx))

        prot_ctx, _ = self.prot_to_drug_attn(
            query=aa_dense, key=atom_dense, value=atom_dense, key_padding_mask=atom_padding_mask
        )
        prot_ctx = self.prot_norm1(aa_dense + prot_ctx)
        prot_ctx = self.prot_norm2(prot_ctx + self.prot_ffn(prot_ctx))

        atom_mask_expanded = atom_mask.unsqueeze(-1).float()
        aa_mask_expanded = aa_mask.unsqueeze(-1).float()

        drug_graph_global = (drug_ctx * atom_mask_expanded).sum(dim=1) / atom_mask_expanded.sum(dim=1).clamp(min=1e-9)
        prot_graph_global = (prot_ctx * aa_mask_expanded).sum(dim=1) / aa_mask_expanded.sum(dim=1).clamp(min=1e-9)

        final_feat = torch.cat([
            drug_graph_global, 
            prot_graph_global, 
            drug_seq_feat, 
            prot_seq_feat
        ], dim=-1)

        scores = self.out_mlp(final_feat)

        if return_attention:
            pocket_weights = drug_to_prot_weights.mean(dim=1) 
            return scores, pocket_weights

        return scores


class FineGrainedCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(FineGrainedCrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, graph_x, graph_batch, seq_x, seq_mask):
        
        graph_dense, graph_mask = to_dense_batch(graph_x, graph_batch)
        Q = self.graph_proj(graph_dense)

        K = V = self.seq_proj(seq_x)

        key_padding_mask = (seq_mask == 0)

        attn_out, _ = self.cross_attn(
            query=Q, 
            key=K, 
            value=V, 
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(Q + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))

        mask_expanded = graph_mask.unsqueeze(-1).float()
        sum_pooled = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        global_repr = sum_pooled / count
        
        return global_repr


class MLP(nn.Module):
    def __init__(self, layers, out_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.out_norm = out_norm
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
        if out_norm:
            self.norm = nn.LayerNorm(layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.out_norm:
            x = self.norm(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedFusion, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, graph_feat, seq_feat):
        concat_feat = torch.cat([graph_feat, seq_feat], dim=-1)
        
        z = self.gate_net(concat_feat)
        
        fused = z * graph_feat + (1 - z) * seq_feat
        
        return fused

class SCDTI(nn.Module):
    def __init__(self, depth=3, device='cuda:0', chemberta_path='./ChemBERTa-zinc-base-v1'):
        super(SCDTI, self).__init__()

        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280
        self.hidden_channels = 200
        self.depth = depth
        self.device = device
        self.drug_gate = GatedFusion(self.hidden_channels)
        self.prot_gate = GatedFusion(self.hidden_channels)

        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        self.bond_encoder = Embedding(10, self.hidden_channels)
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        self.blocks = nn.ModuleList([SCBlock(dropout=0.1) for _ in range(depth)])

        self.drug_encoder = ChemBERTaEncoder(
            model_path=chemberta_path, 
            output_dim=self.hidden_channels,
            freeze=True 
        )
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)
        
        self.blocks_1D = nn.ModuleList([SCBlock_1D(dropout=0.5) for _ in range(depth)])

        self.fine_grained_fusion = FineGrainedCrossAttention(
            hidden_dim=self.hidden_channels,
            num_heads=4,
            dropout=0.1
        )

        self.node_interaction = AtomResidueInteraction_Optimized(
            hidden_dim=self.hidden_channels,
            num_heads=4,
            dropout=0.2
        )

        self.cl_projector = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, 64) 
        )

        self.to(device)

    def forward(self, data, return_features=False, return_attention=False):
        atom_x, atom_x_feat, atom_edge_index, bond_x = \
            data.mol_x, data.mol_x_feat, data.mol_edge_index, data.mol_edge_attr
        
        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight
        
        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch
        m2p_edge_index = data.m2p_edge_index

        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
        bond_x = self.bond_encoder(bond_x)
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)
        
        try:
            from layers import rbf
            aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)
        except:
            aa_edge_attr = None

        drug_repr_graph = []
        prot_repr_graph = []

        for i in range(self.depth):
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, 
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, 
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            
            drug_repr_graph.append(drug_global_repr)
            prot_repr_graph.append(prot_global_repr)

        if hasattr(data, 'smiles'): raw_smiles_list = data.smiles
        elif hasattr(data, 'raw_smiles'): raw_smiles_list = data.raw_smiles
        else: raw_smiles_list = [s[0] if isinstance(s, list) else s for s in data.smiles]

        atom_x_seq, seq_mask = self.drug_encoder(raw_smiles_list, self.device)
        aa_x_seq = self.prot_seq_emb(seq_x)

        drug_repr_seq = []
        prot_repr_seq = []
        
        final_prot_seq_pool = None

        for i in range(self.depth):
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq_updated, aa_x_seq_updated, drug_seq_pool, prot_seq_pool = out_seq
            
            atom_x_seq = atom_x_seq_updated
            aa_x_seq = aa_x_seq_updated
            
            drug_repr_seq.append(drug_seq_pool)
            prot_repr_seq.append(prot_seq_pool)
            final_prot_seq_pool = prot_seq_pool

        drug_fused_global = self.fine_grained_fusion(
            graph_x=atom_x,         
            graph_batch=atom_batch,
            seq_x=atom_x_seq,       
            seq_mask=seq_mask       
        )

        drug_graph_stack = torch.stack(drug_repr_graph, dim=-2)
        drug_graph_pool = torch.mean(drug_graph_stack, dim=1) 
        
        prot_graph_stack = torch.stack(prot_repr_graph, dim=-2)
        prot_graph_pool = torch.mean(prot_graph_stack, dim=1) 

        drug_final = self.drug_gate(drug_graph_pool, drug_fused_global)
        prot_final = self.prot_gate(prot_graph_pool, final_prot_seq_pool)

        if return_attention:
            scores, pocket_weights = self.node_interaction(
                atom_x=atom_x, aa_x=aa_x, atom_batch=atom_batch, aa_batch=aa_batch,
                drug_seq_feat=drug_final, prot_seq_feat=prot_final,
                return_attention=True
            )
            return scores, pocket_weights
        else:
            scores = self.node_interaction(
                atom_x=atom_x, aa_x=aa_x, atom_batch=atom_batch, aa_batch=aa_batch,
                drug_seq_feat=drug_final, prot_seq_feat=prot_final
            )

        if return_features:
            drug_seq_stack = torch.stack(drug_repr_seq, dim=-2)
            prot_seq_stack = torch.stack(prot_repr_seq, dim=-2)

            drug_feat_pool = torch.mean(drug_graph_stack + drug_seq_stack, dim=1)
            prot_feat_pool = torch.mean(prot_graph_stack + prot_seq_stack, dim=1)

            drug_feat_proj = self.cl_projector(drug_feat_pool)
            prot_feat_proj = self.cl_projector(prot_feat_pool)

            return scores, drug_feat_proj, prot_feat_proj

        return scores

def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):
    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0) 
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list
```