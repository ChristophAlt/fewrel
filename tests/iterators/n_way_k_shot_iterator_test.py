from allennlp.data.vocabulary import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from fewrel.dataset_readers import FewRelDatasetReader
from fewrel.iterators import NWayKShotIterator

class TestNWayKShotIterator(AllenNlpTestCase):
    def test_iterate(self):
        BATCH_SIZE = 2
        N = 2
        K = 4
        Q = 2
        NUM_INSTANCES = 9

        reader = FewRelDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/fewrel.json'))

        vocab = Vocabulary.from_instances(instances)

        iterator = NWayKShotIterator(n=N, k=K, q=Q, instances_per_epoch=NUM_INSTANCES, batch_size=BATCH_SIZE)
        iterator.index_with(vocab)

        generator = iterator(instances, shuffle=None, num_epochs=1)

        batch = next(generator)

        num_batches = 1
        for _ in generator:
            num_batches += 1

        assert num_batches == 5
        assert all([field in batch for field in ['support', 'query', 'label']])

        support_tokens_tensor = batch['support']['tokens']
        assert support_tokens_tensor.shape[:2] == (BATCH_SIZE, N*K)

        query_tokens_tensor = batch['query']['tokens']
        assert query_tokens_tensor.shape[:2] == (BATCH_SIZE, N*Q)

        label_tensor = batch['label']
        assert label_tensor.shape[:2] == (BATCH_SIZE, N*Q)

        print('Batch:', batch)
        assert 1 == 0
