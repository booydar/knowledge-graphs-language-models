import json
import logging
import os
import pandas as pd
import shutil
from pathlib import Path
from pymongo import MongoClient

from megatron.data.dataset_utils import get_indexed_dataset_

import horovod.torch as hvd
from dotenv import load_dotenv
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, accuracy_score

from lm_experiments_tools import Trainer, TrainerArgs
from torch.utils.data import Dataset
import datasets

from lm_experiments_tools.hits_calculator import HitsCalculator

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, T5Tokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name, get_optimizer  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='task name')
parser.add_argument('--train_path', type=str, default='.', help='path to train csv')
parser.add_argument('--valid_path', type=str, default='.', help='path to valid csv')
parser.add_argument('--test_path', type=str, default='.', help='path to test csv')

parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=2,
                    help='how many valid examples to show during training (default: 0)')

parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                   'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--drop_neighborhood', action='store_true', default=False, 
                    help='not to include neighborhood in model input')
parser.add_argument('--drop_description', action='store_true', default=False, 
                    help='not to learn to predict entity description')

parser.add_argument('--index_path', default=None, type=str, 
                    help='path to index for hits metric')


# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
## 
parser.add_argument('--cpt_path', type=str, help='path of checkpoint folder')

parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


class KGLMLocalDataset(Dataset):
    def __init__(self, path, neighborhood=True, description=True, sep='[SEP]', sep2=' [SEP-2] '):
        self.df = pd.read_csv(path)
        self.neighborhood = neighborhood
        self.description = description
        self.sep = sep
        self.sep2 = sep2

    def  __getitem__(self, idx):
        item = {}
        triplet = self.df.iloc[idx]
        item["input"] = triplet.verbalization
        if not self.neighborhood:
            item["input"] = self.drop_neighborhood(item["input"])
        
        item["outputs"] = triplet['verbalized_tail']
        if not self.description:
            item["outputs"] = self.drop_description(item["outputs"])
            
        item["output_id"] = triplet["tail"]
            
        return item
        
    def __len__(self):
        return self.df.shape[0]
            
    
    def drop_neighborhood(self, text):
        return self.sep.join(text.split(self.sep)[:2]) + self.sep
    
    
    def drop_description(self, text):
        return text.split(self.sep2)[0]
    

if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # create model path and save configuration
    if hvd.rank() == 0 and args.model_path is not None:
    # if args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)

    # if args.tokenizer:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if args.tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.from_pretrained)

    # add sep token
    # tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    tokenizer.add_tokens(['[SEP]', '[SEP-2]'])
    
    if args.model_type == 'encoder-decoder':
        global_attention_first_token = False  # should be True for LED
        encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        # generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        generate_kwargs = {}

        def collate_fn(batch):
            # print('batch', batch[0].keys(), batch[0]['input'])
            # cut too long strings because they may slow down tokenization
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            if 'outputs' in batch[0]:
                # if we have more than 1 label per example (only in valid) take only one of them
                # to compute loss on valid
                labels = [b['outputs'][:args.target_seq_len * 10] for b in batch]
            else:
                labels = [b['output'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len, return_tensors='pt',
                                                   **encode_plus_kwargs)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len, return_tensors='pt',
                                                     **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels
            if 'outputs' in batch[0]:
                features['target_text'] = [b['outputs'] for b in batch]
            else:
                features['target_text'] = [b['output'] for b in batch]
            if 'output_id' in batch[0]:
                features['output_id'] = [b['output_id'] for b in batch]
            if 'global_attention_mask' in features:
                raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features

    elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
        if args.use_generate_on_valid:
            raise RuntimeError('use_generate_on_valid should be set to False for encoder-only models')

        encode_plus_kwargs = {'max_length': args.input_seq_len,
                              'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 64}
        generate_kwargs = {}
        labels_map = {'Contradiction': 0, 'Entailment': 1, 'Not mentioned': 2}
        num_labels = len(labels_map)

        def collate_fn(batch):
            # cut too long strings because they may slow down tokenization
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            labels = [b['output'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs)
            labels = np.array([labels_map[t] for t in labels])
            features['labels'] = torch.from_numpy(labels)
            return features

    else:
        raise NotImplementedError('only encoder-decoder models are supported for scrolls datasets or '
                                  'encoder models only for contract_nli task')

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for: {args.task_name}')

    train_dataset = KGLMLocalDataset(args.train_path, neighborhood=not args.drop_neighborhood, description=not args.drop_description)
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    # per_worker_batch_size = args.batch_size
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
    # train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,
    #                               collate_fn=collate_fn, **kwargs)
    # get validation dataset
    valid_dataloader = None
    if hvd.rank() == 0:
        logger.info(f'preparing validation data from: {args.task_name}')
    valid_dataset = KGLMLocalDataset(args.valid_path, neighborhood=not args.drop_neighborhood, description=not args.drop_description)

    valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                  collate_fn=collate_fn, **kwargs)
    
    # gen = iter(train_dataloader)
    # print('\n\n\nTrain sample: ', next(gen))
    # print('\nDataset sample: ', train_dataset[10])
    # raise(ValueError)
    test_dataset = KGLMLocalDataset(args.test_path, neighborhood=not args.drop_neighborhood, description=not args.drop_description)
    test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                  collate_fn=collate_fn, **kwargs)
    
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        if args.model_type == 'encoder' and args.task_name == 'contract_nli':
            model_cfg.num_labels = num_labels
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        if args.model_type == 'encoder-decoder':
            model = model_cls.from_pretrained(args.from_pretrained)
        elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
            model = model_cls.from_pretrained(args.from_pretrained, num_labels=num_labels)

    ## load cpt
    if args.cpt_path:
        model_cpt = os.path.join(args.cpt_path, "model_best.pth")
        cpt = torch.load(model_cpt)
        model.load_state_dict(cpt['model_state_dict'])
    
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        if 'generation_outputs' in output:
            data['labels'] = batch['target_text']
            data['generation_outputs'] = output['generation_outputs']
        if args.model_type == 'encoder':
            data['labels'] = batch['labels']
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        if 'output_id' in batch:
            data['output_id'] = batch['output_id']
        return data


    hits_calculator = HitsCalculator(drop_description=args.drop_description, index_path=args.index_path)
    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None

        if args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
            # replace -100 with pad token in labels
            y = data['labels']
            y_entity = list(map(lambda x: x.split("[SEP-2]")[0].strip(), y))
            # print('!', data['generation_outputs'].shape)
            p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
            p_entity = list(map(lambda x: x.split("[SEP-2]")[0].strip(), p))
            if hvd.rank() == 0 and args.show_valid_examples > 0:
            # if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i]}')
                    logger.info(f'p: {p[i]}')
                    logger.info(f'p: {p_entity[i]}')
                    logger.info(f'p ids: {data["generation_outputs"][i]}')
                    logger.info('-' * 50)
            # todo: do we need to better clean P to remove tokens after eos? not remove special tokens only
        elif args.model_type == 'encoder':
            y, p = data['labels'], data['predictions']

        if y is not None and p is not None:
            metrics['exact_match'] = accuracy_score(y, p) * 100
            metrics['exact_match_entity'] = accuracy_score(y_entity, p_entity) * 100

            hits = hits_calculator.hits(p, data['output_id'])
            if hvd.rank() == 0:
                logger.info(f'hits: {hits}')
            for key in hits:
                metrics[key] = hits[key]

        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      generate_kwargs=generate_kwargs if args.use_generate_on_valid else {})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        hvd.barrier()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            if hvd.rank() == 0:
                logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if test_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        # run validation, do not write to tensorboard
        # if hvd.rank() == 0:
        #     logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=False)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
            
        if test_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Running validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
