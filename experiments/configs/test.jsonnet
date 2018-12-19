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
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": false
      },
      "offset_head": {
        "type": "embedding",
        "embedding_dim": 10,
        "trainable": true
      },
      "offset_tail": {
        "type": "embedding",
        "embedding_dim": 10,
        "trainable": true
      },
    },
    "support_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 120,
      "hidden_size": 10,
      "num_layers": 1,
      "dropout": 0.2
    },
    "few_shot_model": "proto"
    // "few_shot_model": {
    //   "type": "proto",
    //   "hidden_dim": 20
    // },
  },
  "iterator": {
    "type": "n_way_k_shot",
    "batch_size": 4,
    "n": 10,
    "k": 5,
    "q": 2,
    "instances_per_epoch": 3000
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}