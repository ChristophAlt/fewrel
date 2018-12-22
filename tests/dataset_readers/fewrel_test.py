from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from fewrel.dataset_readers import FewRelDatasetReader


class TestFewRelDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        MAX_LEN = 100

        reader = FewRelDatasetReader(max_len=MAX_LEN)
        instances = ensure_list(reader.read("tests/fixtures/fewrel.json"))

        instance1 = {
            "tokens": [
                "Merpati",
                "flight",
                "106",
                "departed",
                "Jakarta",
                "(",
                "CGK",
                ")",
                "on",
                "a",
                "domestic",
                "flight",
                "to",
                "Tanjung",
                "Pandan",
                "(",
                "TJQ",
                ")",
                ".",
            ],
            "head": (16, 16),
            "tail": (13, 14),
            "label": "P931",
        }

        assert len(instances) == 20
        fields = instances[0].fields
        tokens = fields["text"].tokens
        assert [t.text for t in tokens] == instance1["tokens"]

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
        head_offsets = [o + MAX_LEN for o in head_offsets]
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
        tail_offsets = [o + MAX_LEN for o in tail_offsets]

        assert all([hasattr(token, "offset_head") for token in tokens])
        assert all([hasattr(token, "offset_tail") for token in tokens])

        assert head_offsets == [token.offset_head for token in tokens]
        assert tail_offsets == [token.offset_tail for token in tokens]
