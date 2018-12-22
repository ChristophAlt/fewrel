from allennlp.data.vocabulary import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from fewrel.dataset_readers import FewRelDatasetReader
from fewrel.token_indexers import OffsetTokenIndexer


class TestOffsetTokenIndexer(AllenNlpTestCase):
    def test_read_from_file(self):
        MAX_LEN = 100
        OFFSET_INDICES_HEAD_NAME = "offset_indices_head"
        OFFSET_INDICES_TAIL_NAME = "offset_indices_tail"

        reader = FewRelDatasetReader(max_len=MAX_LEN)
        instances = ensure_list(reader.read("tests/fixtures/fewrel.json"))

        vocab = Vocabulary.from_instances(instances)

        fields = instances[0].fields
        tokens = fields["text"].tokens

        head_offsets = [
            -16,
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
        ]
        offset_indices_head = {
            OFFSET_INDICES_HEAD_NAME: [o + MAX_LEN for o in head_offsets]
        }

        tail_offsets = [
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            0,
            1,
            2,
            3,
            4,
        ]
        offset_indices_tail = {
            OFFSET_INDICES_TAIL_NAME: [o + MAX_LEN for o in tail_offsets]
        }

        token_indexer_head = OffsetTokenIndexer(token_attribute="offset_head")
        token_indexer_tail = OffsetTokenIndexer(token_attribute="offset_tail")

        assert offset_indices_head == token_indexer_head.tokens_to_indices(
            tokens, vocab, OFFSET_INDICES_HEAD_NAME
        )

        assert offset_indices_tail == token_indexer_tail.tokens_to_indices(
            tokens, vocab, OFFSET_INDICES_TAIL_NAME
        )
