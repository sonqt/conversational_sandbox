from utils import parseargs

args = parseargs([['-model', '--model_name_or_path', 'model_name', str],
                     ['-corpus', '--corpus_name', 'corpus_name', str],
                     ['-train', '--do_train', 'train_before_evaluate', bool],
                     ['-eval', '--do_eval', 'evaluate_or_not', bool],
                     ['-lr', '--learning_rate', 'learning_rate', float, 2e-5],
                     ['-bs', '--per_device_batch_size', 'number_of_samples_on_each_GPU', int, 8],
                     ['-epoch', '--num_train_epochs', 'num_train_epochs', int, 5],
                     ['-output', '--output_dir', 'output_directory', str],
                     ['-seed', '--random_seed', 'random_seed', int, 42]
                     ])
                     # type_context
print(f'ARGPARSE OPTIONS {args}')
if args.do_train:
    print("Will train soon")
if args.do_eval:
    print("Evaluate")
print(type(args.num_train_epochs))