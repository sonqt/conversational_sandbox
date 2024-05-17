# Dynamic Forecasting of Conversation Derailment (Kementchedjhieva and Søgaard, 2021)

I reimplement the derailment forecasting model described in (Kementchedjhieva and Søgaard, 2021).

Command line:
```
MODEL="<model name from hugging face hub here>"
data="<name of the corpus (must be cmv or wikiconv)>"
for seed in 1 2 3 4 5 6 7 8 9 10
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
        --output_dir "<Your output directory here>/${data}/${MODEL}/seed-${seed}"
done
```

