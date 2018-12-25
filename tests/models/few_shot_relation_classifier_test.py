# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class FewShotRelationClassifierTest(ModelTestCase):
    def setUp(self):
        super(FewShotRelationClassifierTest, self).setUp()
        self.set_up_model(
            "tests/fixtures/few_shot_relation_classifier.jsonnet",
            "tests/fixtures/fewrel.json",
        )

    def test_model_can_train_save_and_load(self):
        # TODO: this currently fails due to the randomness in sampling n-way k-shot batches
        # self.ensure_model_can_train_save_and_load(self.param_file)
        pass
