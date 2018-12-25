from typing import Dict, List, Tuple
import logging
import codecs

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("trec")
class TRECDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples from the TREC dataset, and creates a
    dataset suitable for question type classification.
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
        Defaults to ``JustSpacesWordSplitter()``.
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
        self._tokenizer = tokenizer or JustSpacesWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with codecs.open(
            cached_path(file_path), "r", encoding="utf-8", errors="ignore"
        ) as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                question_type, text = line.strip().split(" ", 1)
                if not self._fine_grained:
                    question_type, _ = question_type.split(":")
                yield self.text_to_instance(text, question_type)

    @overrides
    def text_to_instance(
        self, text: str, question_type: str = None
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text = self._tokenizer.split_words(text)

        text_tokens_field = TextField(tokenized_text, self._token_indexers)
        fields = {"text": text_tokens_field}
        if question_type is not None:
            fields["label"] = LabelField(question_type)
        return Instance(fields)
