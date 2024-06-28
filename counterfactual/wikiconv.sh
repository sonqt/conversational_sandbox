MODEL="roberta-base"
data="wikiconv"
for seed in 1 2 3 4
do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python forecast.py\
        --model_name_or_path ${MODEL}\
        --corpus_name ${data}\
        --do_train True\
        --do_eval True\
        --learning_rate 6.7e-6\
        --per_device_batch_size 4\
        --num_train_epochs 5\
        --random_seed ${seed}\
        --output_dir "/reef/sqt2/BERTCRAFT_counterfactual/${data}/${MODEL}/seed-${seed}"
done
