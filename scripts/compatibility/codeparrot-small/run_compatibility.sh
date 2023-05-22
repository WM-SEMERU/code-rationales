export HF_DATASETS_CACHE=/workspaces/code-rationales/datax/hf_datasets_cache
MODEL=codeparrot/codeparrot-small
#DATA_DIR=/workspaces/code-rationales/sequential-rationales/compatibility/gpt2-medium/data
CHECKPOINT_DIR=/workspaces/code-rationales/data/codeparrot-small/checkpoints
LOGGING_DIR=/workspaces/code-rationales/scripts/compatibility/codeparrot-small/logs
nohup python3 -u /workspaces/code-rationales/sequential-rationales/huggingface/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --dataset_name codeparrot/codeparrot-clean \
    --logging_dir $LOGGING_DIR \
    --output_dir $CHECKPOINT_DIR \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 500 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00001 \
    --block_size 512 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 45000 \
    --word_dropout_mixture 0.5 \
    > /workspaces/code-rationales/scripts/compatibility/codeparrot-small/logs/output.txt &