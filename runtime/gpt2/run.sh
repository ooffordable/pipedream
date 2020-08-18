python -m launch --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node=6 \
        main_with_runtime.py \
        --master_addr localhost \
        --module models.depth=8 \
        --train_batch_size 1 \
        --train_data_file /home/soojeong/wikipedia_data/train.json \
        --do_train \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path tests/depth=8/conf6.json --recompute_step
