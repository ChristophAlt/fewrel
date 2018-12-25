from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from fewrel.dataset_readers import TRECDatasetReader


class TestFewRelDatasetReader(AllenNlpTestCase):
    def test_read_from_file_fine_grained(self):
        reader = TRECDatasetReader(fine_grained=True)
        instances = ensure_list(reader.read("tests/fixtures/trec.label"))

        instance1 = {
            "text": [
                "How",
                "did",
                "serfdom",
                "develop",
                "in",
                "and",
                "then",
                "leave",
                "Russia",
                "?",
            ],
            "question_type": "DESC:manner",
        }

        instance2 = {
            "text": [
                "What",
                "films",
                "featured",
                "the",
                "character",
                "Popeye",
                "Doyle",
                "?",
            ],
            "question_type": "ENTY:cremat",
        }

        assert len(instances) == 5

        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens] == instance1["text"]
        assert fields["label"].label == instance1["question_type"]

        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens] == instance2["text"]
        assert fields["label"].label == instance2["question_type"]

    def test_read_from_file(self):
        reader = TRECDatasetReader(fine_grained=False)
        instances = ensure_list(reader.read("tests/fixtures/trec.label"))

        instance1 = {
            "text": [
                "How",
                "did",
                "serfdom",
                "develop",
                "in",
                "and",
                "then",
                "leave",
                "Russia",
                "?",
            ],
            "question_type": "DESC",
        }

        instance2 = {
            "text": [
                "What",
                "films",
                "featured",
                "the",
                "character",
                "Popeye",
                "Doyle",
                "?",
            ],
            "question_type": "ENTY",
        }

        assert len(instances) == 5

        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens] == instance1["text"]
        assert fields["label"].label == instance1["question_type"]

        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens] == instance2["text"]
        assert fields["label"].label == instance2["question_type"]
