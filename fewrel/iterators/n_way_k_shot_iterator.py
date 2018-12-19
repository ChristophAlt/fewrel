import logging
import random
from collections import deque
from itertools import groupby
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


@DataIterator.register("n_way_k_shot")
class NWayKShotIterator(DataIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).
    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.
        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.
        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    """

    def __init__(self,
                 n: int,
                 k: int,
                 q: int,
                 instances_per_epoch: int,
                 text_field: str='text',
                 label_field: str='label',
                 #sorting_keys: List[Tuple[str, str]],
                 #padding_noise: float = 0.1,
                 #biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        # if not sorting_keys:
        #     raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._n = n
        self._k = k
        self._q = q
        self._text_field = text_field
        self._label_field = label_field
        #self._sorting_keys = sorting_keys
        #self._padding_noise = padding_noise
        #self._biggest_batch_first = biggest_batch_first

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        def index_with_list(l, index):
            return [l[i] for i in index]

        # TODO: handle case when instances do not fit in memory (for now, assume we have access to all instances at once).
        for instance_list in self._memory_sized_lists(instances):

            # group instances by label
            # sample n classes
            # from the n classes, sample k + q instances
            # create a batch of
            whole_division = {}
            grouping_func = lambda inst: inst['label'].label
            for label, grouped_instances in groupby(sorted(instances, key=grouping_func), key=grouping_func):
                whole_division[label] = list(grouped_instances)

            N = self._n
            K = self._k
            Q = self._q
            labels = whole_division.keys()

            n_way_k_shot_instances = []
            for _ in range(self._instances_per_epoch):
                all_support_instances = []
                all_query_instances = []
                all_labels = []

                target_labels = random.sample(labels, N)
                for idx, label in enumerate(target_labels):
                    label_instances = whole_division[label]
                    indices = np.random.choice(len(label_instances), K + Q, replace=False)
                    selected_instances = index_with_list(label_instances, indices)
                    support_instances, query_instances = selected_instances[:K], selected_instances[K:]
                    all_support_instances += support_instances
                    all_query_instances += query_instances
                    all_labels += [idx] * Q

                indices_perm = np.random.permutation(N * Q)
                all_query_instances = index_with_list(all_query_instances, indices_perm)
                all_labels = index_with_list(all_labels, indices_perm)

                fields = {
                    'support': ListField([inst[self._text_field] for inst in all_support_instances]),
                    'query': ListField([inst[self._text_field] for inst in all_query_instances]),
                    'label': ListField([LabelField(idx, skip_indexing=True) for idx in all_labels])
                }
                n_way_k_shot_instances.append(Instance(fields))

            instance_list = n_way_k_shot_instances
            # instance_list = sort_by_padding(instance_list,
            #                                 self._sorting_keys,
            #                                 self.vocab,
            #                                 self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))

            # move_to_front = self._biggest_batch_first and len(batches) > 1
            # if move_to_front:
            #     # We'll actually pop the last _two_ batches, because the last one might not be full.
            #     last_batch = batches.pop()
            #     penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            # if move_to_front:
            #     batches.insert(0, penultimate_batch)
            #     batches.insert(0, last_batch)

            yield from batches
