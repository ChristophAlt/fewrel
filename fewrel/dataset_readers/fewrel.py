from typing import Dict, List, Tuple
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fewrel")
class FewRelDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples from the FewRel dataset, and creates a
    dataset suitable for few-shot relation classification.
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        head: ``SpanField``
        tail: ``SpanField``
        label: ``LabelField``
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            data = json.load(data_file)
            for relation, examples in data.items():
                for example in examples:
                    tokens = example['tokens']
                    head_indices = example['h'][2][0]
                    head = (head_indices[0], head_indices[-1])
                    tail_indices = example['t'][2][0]
                    tail = (tail_indices[0], tail_indices[-1])
                    yield self.text_to_instance(tokens, head, tail, relation)

    @overrides
    def text_to_instance(self, text_tokens: List[str], head: Tuple[int, int], tail: Tuple[int, int],
                         relation: str=None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        # TODO: maybe support non-tokenized text input
        text_tokens_field = TextField([Token(t) for t in text_tokens], self._token_indexers)
        head_field = SpanField(head[0], head[1], text_tokens_field)
        tail_field = SpanField(tail[0], tail[1], text_tokens_field)
        fields = {
                'text': text_tokens_field,
                'head': head_field,
                'tail': tail_field
            }
        if relation is not None:
            fields['label'] = LabelField(relation)
        return Instance(fields)
