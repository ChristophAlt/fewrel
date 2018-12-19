import torch
from allennlp.models.model import Model


@Model.register('proto')
class Proto(torch.nn.Module):
    
    def __init__(self, hidden_dim: int):
        super(Proto, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set. [B x N x K, input_dim]
        query: Inputs of the query set. [B x N x Q, input_dim]
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = support.view(-1, N, K, self.hidden_dim) # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_dim) # (B, N * Q, D)
         
        # Prototypical Networks 
        support = torch.mean(support, 2) # Calculate prototype for each class
        logits = -self.__batch_dist__(support, query)
        return logits
