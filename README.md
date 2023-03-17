## Domain Adaptation for Enumeration-aware Molecular Transformers 

Install depedencies
`bash
pip install -r requirements.txt
`

Run training<br/>
`bash
    python train.py \
    --model_name_or_path shahrukhx01/smole-bert \
    --train_file data/augmented_hips_2x_simcse.csv \
    --output_dir result/mlm-mtr-simcse-hips \
    --normalization_path data/hips_normalization_values_207.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 100000 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_mtr \
    --do_mlm \
    --cl_weight 0.33\
    --mlm_weight 0.33\
    --mtr_weight 0.33 \
    "$@"
`


ems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## References
Code base adapted from:
* [SimCSE](https://github.com/princeton-nlp/SimCSE)
* [Chemberta-2](https://github.com/seyonechithrananda/bert-loves-chemistry)
