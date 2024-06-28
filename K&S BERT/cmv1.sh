MODEL="bert-base-cased"
data="cmv"
for seed in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python forecast.py\
        --model_name_or_path ${MODEL}\
        --corpus_name ${data}\
        --do_train True\
        --do_eval True\
        --learning_rate 6.7e-6\
        --per_device_batch_size 4\
        --num_train_epochs 7\
        --random_seed ${seed}\
        --output_dir "/reef/sqt2/BERTCRAFT_adversarial/${data}/${MODEL}/seed-${seed}"
done
