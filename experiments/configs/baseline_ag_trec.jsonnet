{
  "dataset_reader": {
    "type": "ag_news",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
  },
  "validation_dataset_reader": {
    "type": "trec",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
  },
  "train_data_path": "./data/ag_news/test.csv",
  "validation_data_path": "./data/trec/TREC_10.label",
  "model": {
    "type": "few_shot_relation_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        "embedding_dim": 50,
        "trainable": false
      },
    },
    "support_encoder": {
      "type": "cnn",
      "embedding_dim": "50",
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
    "n": 2,
    "k": 5,
    "q": 5,
    "instances_per_epoch": 3000
  },
  "validation_iterator": {
    "type": "n_way_k_shot",
    "batch_size": 4,
    "n": 6,
    "k": 3,
    "q": 2,
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