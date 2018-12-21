{
  "dataset_reader": {
    "type": "fewrel",
    "max_len": 100,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "offset_head": {
        "type": "offset",
        "token_attribute": "offset_head"
      },
      "offset_tail": {
        "type": "offset",
        "token_attribute": "offset_tail"
      }
    },
  },
  "train_data_path": "./tests/fixtures/fewrel.json",
  "validation_data_path": "./tests/fixtures/fewrel.json",
  "model": {
    "type": "few_shot_relation_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 2,
        "trainable": false
      },
      "offset_head": {
        "type": "embedding",
        "embedding_dim": 2,
        "trainable": true
      },
      "offset_tail": {
        "type": "embedding",
        "embedding_dim": 2,
        "trainable": true
      },
    },
    "support_encoder": {
      "type": "cnn",
      "embedding_dim": 6,
      "num_filters": 2,
      "conv_layer_activation": "sigmoid",
      "ngram_filter_sizes": [3]
    },
    "few_shot_model": {
      "type": "prototypical_network",
      "hidden_dim": 2
    },
  },
  "iterator": {
    "type": "n_way_k_shot",
    "batch_size": 2,
    "n": 2,
    "k": 2,
    "q": 2,
    "instances_per_epoch": 1
  },
  "validation_iterator": {
    "type": "n_way_k_shot",
    "batch_size": 2,
    "n": 2,
    "k": 2,
    "q": 2,
    "instances_per_epoch": 1
  },
  "trainer": {
    "num_epochs": 1,
    "patience": 5,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}
