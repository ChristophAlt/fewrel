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
  "train_data_path": "./data/fewrel_train.json",
  "validation_data_path": "./data/fewrel_val.json",
  "model": {
    "type": "few_shot_relation_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        "embedding_dim": 50,
        "trainable": false
      },
      "offset_head": {
        "type": "embedding",
        "embedding_dim": 5,
        "trainable": true
      },
      "offset_tail": {
        "type": "embedding",
        "embedding_dim": 5,
        "trainable": true
      },
    },
    "support_encoder": {
      "type": "cnn",
      "embedding_dim": "60",
      "num_filters": 230,
      "ngram_filter_sizes": [3]
    },
    "few_shot_model": {
      "type": "prototypical_network",
      "hidden_dim": 230
    },
  },
  "iterator": {
    "type": "n_way_k_shot",
    "batch_size": 4,
    "n": 20,
    "k": 5,
    "q": 5,
    "instances_per_epoch": 3000
  },
  "validation_iterator": {
    "type": "n_way_k_shot",
    "batch_size": 4,
    "n": 5,
    "k": 5,
    "q": 5,
    "instances_per_epoch": 12000
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 5,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}