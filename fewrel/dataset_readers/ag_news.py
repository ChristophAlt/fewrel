from typing import Dict, List, Tuple
import logging
import csv
import re

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ag_news")
class AGNewsDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples from the AG News dataset, and creates a
    dataset suitable for ? classification.
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        label: ``LabelField``
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the text into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(
        self,
        lazy: bool = False,
        fine_grained=False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy)
        self._fine_grained = fine_grained
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in csv.reader(data_file, quotechar='"', delimiter=","):
                label = line[0]
                text = ""
                for s in line[1:]:
                    text = (
                        text
                        + " "
                        + re.sub(r"^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                    )
                yield self.text_to_instance(text, label)

    @overrides
    def text_to_instance(
        self, text: str, label: str = None
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text = self._tokenizer.tokenize(text)

        text_tokens_field = TextField(tokenized_text, self._token_indexers)
        fields = {"text": text_tokens_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)
