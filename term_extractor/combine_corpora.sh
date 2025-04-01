python preprocess_acter.py \
    --input_dirs /content/ACTER/en/corp/annotated/annotations/sequential_annotations/iob_annotations/without_named_entities/ \
    /content/ACTER/en/equi/annotated/annotations/sequential_annotations/iob_annotations/without_named_entities/ \
    /content/ACTER/en/htfl/annotated/annotations/sequential_annotations/iob_annotations/without_named_entities/ \
    /content/ACTER/en/wind/annotated/annotations/sequential_annotations/iob_annotations/without_named_entities/ \
    --train_output train_full.tsv \
    --test_output test_full.tsv \
    --val_output val_full.tsv \
    --test_size 0.2 \
    --val_size 0 \
    --seed 42