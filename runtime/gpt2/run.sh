python -m launch --nnodes 2 \
        --node_rank 0 \
        --nproc_per_node 4 \
        main_with_runtime.py \
        --master_addr 01.elsa.snuspl.snu.ac.kr \
        --module models.depth=8 \
        --train_batch_size 1 \
        --train_data_file ~/wikitext-2-raw/wiki.train.raw \
        --do_train \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path tests/depth=8/conf.json --recompute_step
