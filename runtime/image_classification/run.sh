python -m launch --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node 4 \
        main_with_runtime.py \
        --master_addr 01.elsa.snuspl.snu.ac.kr \
        --config_path models/vgg16/gpus=16_straight/mp_conf_gpu8.json \
        --module models.vgg16.gpus=16_straight \
        --num_ranks_in_server 4 \
        --b 32 \
        --train_size 2560 \
        --epochs 3
