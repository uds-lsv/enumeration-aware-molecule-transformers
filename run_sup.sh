python train.py \
    --model_name_or_path prajjwal1/bert-tiny \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/my-sup-simcse-bert-mini \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
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
    "$@"