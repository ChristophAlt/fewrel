from typing import Dict, Optional, Any, List

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from fewrel.models.few_shot import FewShotModel


@Model.register("few_shot_relation_classifier")
class FewShotRelationClassifier(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    support_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    query_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    few_shot_model : ``torch.nn.Module``
        The model that predicts the class, based on the encoded query and support.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 few_shot_model: FewShotModel,
                 support_encoder: Seq2VecEncoder,
                 query_encoder: Seq2VecEncoder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(FewShotRelationClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.support_encoder = support_encoder
        self.query_encoder = query_encoder or support_encoder

        if text_field_embedder.get_output_dim() != support_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the support_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            support_encoder.get_input_dim()))
        
        if text_field_embedder.get_output_dim() != self.query_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the query_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            self.query_encoder.get_input_dim()))
        
        self.few_shot_model = few_shot_model

        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                support: Dict[str, torch.LongTensor],
                query: Dict[str, torch.LongTensor],
                metadata: List[Any],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        support : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        query : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # TODO: find a better way to handle this.
        N = metadata[0]['N']
        K = metadata[0]['K']
        Q = metadata[0]['Q']

        # Shape: [B x N x K, seq_len]
        embedded_support = self.text_field_embedder(support)
        # [B x N x K, seq_len, d_emb]
        support_mask = util.get_text_field_mask(support, num_wrapping_dims=1)

        # ====
        _, _, seq_len, d = embedded_support.shape
        embedded_support = embedded_support.view(-1, seq_len, d)
        support_mask = support_mask.view(-1, support_mask.size(-1))
        # ====

        # Shape: [B x N x K, d_enc]
        encoded_support = self.support_encoder(embedded_support, support_mask)

        # Shape: [B x N x Q, seq_len]
        embedded_query = self.text_field_embedder(query)
        # Shape: [B x N x Q, seq_len, d_emb]
        query_mask = util.get_text_field_mask(query, num_wrapping_dims=1)

        # ====
        _, _, seq_len, d = embedded_query.shape
        embedded_query = embedded_query.view(-1, seq_len, d)
        query_mask = query_mask.view(-1, query_mask.size(-1))
        # ====

        # Shape: [B x N x Q, d_enc]
        encoded_query = self.query_encoder(embedded_query, query_mask)

        # Shape: [B, Q, N]
        logits = self.few_shot_model(encoded_support, encoded_query, N, K, Q)

        output_dict = {'logits': logits}
        if label is not None:
            # Shape: [B x Q]
            label = label.view(-1)
            # Shape: [B x Q, N]
            logits = logits.view(-1, logits.size(-1))

            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        output_dict['label'] = argmax_indices.tolist()
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
