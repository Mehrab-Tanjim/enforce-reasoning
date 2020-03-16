#!/bin/bash

NUM_GPUS=$2
bs=$3
num_workers=$4
droot=$5
save_name=$6

if [ $1 = 'debug' ];
then
    echo 'In Debug mode'
    python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 10 --tasks 1 --save_name vilbert-loss-debug --debug

elif [ $1 = 'eval' ];
then
    echo ' Eval mode'
    python eval_tasks_subset.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 2 --tasks 1 --save_name vilbert-var-loss-debug --debug

else
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers $num_workers --tasks 1 --save_name $save_name --batch_size $bs --data_root $droot

fi

# Usage debug train_vcr.sh debug
# Usage train_vcr.sh train 4 64 16 /mnt/dst vlbert-gpt2

############ Eval task #############3
# Without distributed training, batch size of 20 works, and num_workers = 10
# python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 1 --split val --batch_size 20


# With distributed training, even batch size of 2 and num_worker = 1 doesn't work
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0  eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 1 --split val --batch_size 2 --num_workers 1


