import torch
from fewrel.models.few_shot import FewShotModel


@FewShotModel.register('prototypical_network')
class PrototypicalNetwork(FewShotModel):
    
    def __init__(self, hidden_dim: int):
        super(PrototypicalNetwork, self).__init__()
        self.hidden_dim = hidden_dim

    def euclidean_distance(self, x: torch.Tensor, y: torch.Tensor, dim: int):
        return (torch.pow(x - y, 2)).sum(dim)

    def batch_distance(self, prototypes: torch.Tensor, query: torch.Tensor):
        return self.euclidean_distance(prototypes.unsqueeze(1), query.unsqueeze(2), 3)

    def forward(self, support: torch.Tensor, query: torch.Tensor, n: int, k: int, q: int):
        '''
        support: Instances of the support set, of shape [B x N x K, input_dim]
        query: Instances of the query set, of shape [B x N x Q, input_dim]
        n: Number of classes
        k: Number of instances for each class in the support set
        q: Number of instances for each class in the query set
        '''

        # Shape: [B, D, K, d_hidden]
        support = support.view(-1, n, k, self.hidden_dim)
        # Shape: [B, N, Q, d_hidden]
        query = query.view(-1, n * q, self.hidden_dim)
         
        # Calculate prototype for each class
        # Shape: [B, N, d_hidden]
        prototypes = torch.mean(support, 2)
        # Shape: [B, Q, N]
        logits = -1. * self.batch_distance(prototypes, query)
        return logits
