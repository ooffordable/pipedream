python -m launch --dist_world_size 6 \
        --nproc_this_node 6 \
        --num_prev_ranks 0 \
        main_with_runtime.py \
        --master_addr 06.elsa.snuspl.snu.ac.kr \
        --module models.depth=8 \
        --train_batch_size 1 \
        --train_data_file ~/wikitext-2-raw/wiki.train.raw \
        --do_train \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path tests/depth=8/conf6.json --recompute_step --gpipe
