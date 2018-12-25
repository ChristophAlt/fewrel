from typing import Tuple
from itertools import chain

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import ListField, MetadataField
from allennlp.predictors.predictor import Predictor


@Predictor.register("few-shot-relation-classifier")
class FewShotRelationClassifierPredictor(Predictor):
    """"Predictor wrapper for the FewShotRelationClassifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        support_set = json_dict["support"]
        query = json_dict["query"]

        support_instances = [
            self._dataset_reader.text_to_instance(**support)
            for support in chain.from_iterable(support_set)
        ]
        query_instance = self._dataset_reader.text_to_instance(**query)

        # for now, assume each class has the same number of supporting examples
        assert len(set([len(class_support) for class_support in support_set])) == 1

        N = len(support_set)
        K = len(support_set[0])

        fields = {
            "support": ListField([inst["text"] for inst in support_instances]),
            # TODO: change this as soon as few-shot classifier supports an
            # arbitrary number of queries per support set
            "query": ListField(N * [query_instance["text"]]),
            "metadata": MetadataField(dict(N=N, K=K, Q=1)),
        }

        return Instance(fields)
