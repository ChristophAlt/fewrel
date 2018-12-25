# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import fewrel


class TestPaperClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        # TODO: load json example from test fixture (fewrel.json)
        inputs = {}

        # TODO: create a super-small model for loading at test time
        # archive = load_archive("tests/fixtures/model.tar.gz")
        # predictor = Predictor.from_archive(archive, "few-shot-relation-classifier")

        # result = predictor.predict_json(inputs)

        # label = result.get("label")
        # assert 0 <= label < len(inputs["support"])

        # class_probabilities = result.get("class_probabilities")
        # assert class_probabilities is not None
        # assert all(cp > 0 for cp in class_probabilities)
        # assert sum(class_probabilities) == approx(1.0)
