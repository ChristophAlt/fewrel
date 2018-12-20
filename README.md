# Experiment Environment for Few-Shot Relation Extraction

The repo contains additional functionality, not implemented in AllenNLP, for few-shot relation extraction / relation classification.

- Dataset reader for [FewRel](http://zhuhao.me/fewrel) dataset
- N-way K-shot iterator, creating batches according to the few-shot scenario
- Few-shot relation classifier, configurable via jsonnet files (see experiments/config/ for examples)
- Relative offset embeddings (required for non-sequential models, i.e. CNN)
- Additional models for few-shot classification
  - Prototypcial Network

## Setup
`$ pip install -r requirements.txt`

## Run experiment

`$ allennlp train --include-package fewrel experiments/configs/baseline.jsonnet -s <RESULT DIR>`

## References

- Han, Xu et al. [“FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation.”](https://arxiv.org/pdf/1810.10147.pdf) EMNLP (2018).
- Snell, Jake et al. [“Prototypical Networks for Few-shot Learning.”](https://arxiv.org/pdf/1703.05175.pdf) NIPS (2017).
