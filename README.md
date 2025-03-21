# ACTER-NER

This is an assorted set of experiments to convert ATE task to NER task and train the model for an automatic term extraction.

## Prerequisites
You'll need:
1. [ACTER dataset](https://github.com/AylaRT/ACTER/tree/master)
2. Latest spacy
3. Some GPU (ideally).

## Steps to reproduce
1. Checkout the repo.
2. Install fresh spacy
3. Download ACTER files (**en/corp**/annotated/annotations/sequential_annotations/iob_annotations/without_named_entities/) etc. You need sequential annotation in the IOB format without named entities.
4. Check `combine_corpora.sh` script to launch it on the subset of files (currently it combines english texts from all 4 domains). Modify if needed.
5. Run the script with desired data and params, that'll render you a set of train/test/validation tsv files.
6. Convert those files into spacy data format like this: `spacy convert --converter iob train_full.tsv output` where output is a directory. For your convenience, this repo includes converted files.
7. Pick the train config for spacy from the configs folder. Adjust it if needed to point to the correct files or to change base model and hyperparams.
8. Run `spacy train` using given config, consult the spacy docs if needed.
9. You should have your model

The best results I got so far for english language was on `FacebookAI/roberta-large`:

``` JSON
  "performance":{
    "ents_f":0.9329312865,
    "ents_p":0.9348104743,
    "ents_r":0.9310596389,
    "ents_per_type":{
      "TERM":{
        "p":0.9348104743,
        "r":0.9310596389,
        "f":0.9329312865
      }
    },
    "transformer_loss":143.8517941209,
    "ner_loss":186.9935709015
  }
```
