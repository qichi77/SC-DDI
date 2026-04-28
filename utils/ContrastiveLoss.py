```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(GraphContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, drug_features, prot_features, labels):
        drug_features = F.normalize(drug_features, dim=1)
        prot_features = F.normalize(prot_features, dim=1)

        logits = torch.matmul(drug_features, prot_features.T) / self.temperature

        batch_size = labels.shape[0]

        mask = torch.eye(batch_size).to(drug_features.device)
        positive_mask = mask * labels.unsqueeze(1)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-6)

        valid_indices = labels > 0
        if valid_indices.sum() > 0:
            loss = -mean_log_prob_pos[valid_indices].mean()
        else:
            loss = torch.tensor(0.0).to(drug_features.device)

        return loss
```