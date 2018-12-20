import torch
from allennlp.common.registrable import Registrable

from typing import Dict


class FewShotModel(torch.nn.Module, Registrable):
    def forward(self,
                support: Dict[str, torch.Tensor],
                query: Dict[str, torch.Tensor],
                n: int, k: int, q: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
