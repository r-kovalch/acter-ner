# ACTER-NER

This is an assorted set of experiments to convert ATE task to NER task and train the model for an automatic term extraction.

## Fine-tuning results


[//]: # (model, language, ents_f, ents_p, ents_r, score)

[//]: # (xlm-roberta-large, en+fr+nl, 84.80, 86.61, 83.08, 0.85)

[//]: # (xlm-roberta-large, en, 90.51, 90.89, 90.14, 0.91)

[//]: # (roberta-large, en+fr+nl, 85.51, 88.16, 83.01, 0.86)

[//]: # (roberta-large, en, 92.72, 93.12, 92.33, 0.93)
| model                                                                                   | ents_f    | ents_p    | ents_r | score    |
|-----------------------------------------------------------------------------------------|-----------|-----------|-------|----------|
| **English only**                                                                        |           |           |       |          |
| [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)     | 90.51     | 90.89     | 90.14 | 0.91     |
| [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)             | **92.72** | **93.12** | **92.33** | **0.93** |
| [urchade/gliner_small_v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5)  | 81.67     | 83.05     | 80.33 | 0.82     |
| [urchade/gliner_large_v2.5](https://huggingface.co/gliner-community/gliner_large-v2.5)* | 75.00     | 65.06     | 88.52 | 0.75     |
| **Multilingual (English, French, Dutch)**                                               |           |           |       |          |
| [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)     | 84.80     | 86.61     | 83.08 | 0.85     |
| [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)             | **85.51** | **88.16** | 83.01 | **0.86** |
| [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1)           | 84.61     | 84.42     | **84.95** | 0.85     |

---
- Due to GPU and cost limitations, we used L4 GPU on Colab, which is not enough for complete gliner_large or gliner_medium fine-tuning, so the resulting large and medium models might be underfitted.
- For GLiNER models, recommended threshold from the original paper was used - 0.35, adjusting which may yield better F1 score, but at the cost fo reducing recall.
- Hyperparameter tuning may improve the results.

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
