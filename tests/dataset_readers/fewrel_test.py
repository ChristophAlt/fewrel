from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from fewrel.dataset_readers import FewRelDatasetReader

class TestFewRelDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = FewRelDatasetReader()
        instances = ensure_list(reader.read("tests/fixtures/fewrel.json"))

        instance1 = {
            "tokens": [
                "Merpati", "flight", "106", "departed", "Jakarta", "(", "CGK", ")", "on", "a", "domestic",
                "flight", "to", "Tanjung", "Pandan", "(", "TJQ", ")", "."
            ],
            "head": (16, 16),
            "tail": (13, 14),
            "label": "P931"
        }

        assert len(instances) == 20
        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens] == instance1["tokens"]
        head = fields["head"]
        assert (head.span_start, head.span_end) == instance1["head"]
        tail = fields["tail"]
        assert (tail.span_start, tail.span_end) == instance1["tail"]
        assert fields["label"].label == instance1["label"]
