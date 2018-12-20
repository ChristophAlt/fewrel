import torch
from fewrel.models.few_shot import FewShotModel


@FewShotModel.register('prototypical_network')
class PrototypicalNetwork(FewShotModel):
    
    def __init__(self, hidden_dim: int):
        super(PrototypicalNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)

    def distance(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def batch_distance(self, prototypes, query):
        return self.distance(prototypes.unsqueeze(1), query.unsqueeze(2), 3)

    def forward(self, support, query, n, k, q):
        '''
        support: Instances of the support set, of size [B x N x K, input_dim]
        query: Instances of the query set, of size [B x N x Q, input_dim]
        n: Number of classes
        k: Number of instances for each class in the support set
        q: Number of instances for each class in the query set
        '''

        # [B, N, K, d_hidden]
        support = support.view(-1, n, k, self.hidden_dim) # (B, N, K, D)
        # [B, N, Q, d_hidden]
        query = query.view(-1, n * q, self.hidden_dim) # (B, N * Q, D)
         
        # Calculate class prototypes
        prototypes = torch.mean(support, 2)
        logits = -1. * self.batch_distance(prototypes, query)
        return logits
