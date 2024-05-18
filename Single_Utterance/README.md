# Do we need Conversational Dynamics?

MAIN IDEA: some instances require dynamics while for others utterance-level prediction is sufficient.

Utterance Model: A conversation in our dataset has N utterances. We train K utterance models to predict if the Nth utterance contains personal attack given the (N-1)th utterance. 

## Command 
```
MODEL="<model name from hugging face hub here>"
data="<name of the corpus (must be cmv or wikiconv)>"
for seed in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python single_utt.py\
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
    