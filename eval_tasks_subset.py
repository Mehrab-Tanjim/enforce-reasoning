import argparse
import json
import logging
import os
import random
from io import open
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import sacrebleu
import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

# from parallel.parallel import DataParallelModel, DataParallelCriterion

from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from vilbert.optimization import BertAdam, Adam, Adamax
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import vilbert.utils as utils
import torch.distributed as dist
from transformers import GPT2Tokenizer
from vilbert.gpt2_rationale import sample_sequence
from transformers import GPT2Config, GPT2LMHeadModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ViLBertGPT2(nn.Module):

    def __init__(self, vilbert, gpt2_tokenizer, gpt2_embed_dim=768, config=None):
        nn.Module.__init__(self)
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_embed_dim = gpt2_embed_dim
        self.embed = torch.nn.Linear(config.bi_hidden_size, self.gpt2_embed_dim)
        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', from_tf=False, config=self.gpt2_config)
        self.vilbert_model = vilbert

    def forward(self, rationale_text_label, generate, q_id, question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, num_options, freeze=-1):
        outs = self.vilbert_model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, num_options=num_options)

        gpt2_inp, pred_ans = outs[7:]
        gpt2_inp = self.embed(gpt2_inp)
        gpt2_inputs = (gpt2_inp, rationale_text_label)
        gpt2_outputs = self.gpt2_model(gpt2_inputs, labels=rationale_text_label)
        gpt2_loss = gpt2_outputs[0]

        to_return = outs[:7] + (gpt2_loss,)

        references=[]
        hypotheses=[]
        if generate:
            out = sample_sequence(
                    model=self.gpt2_model,
                    context=gpt2_inp,
                    length=30, #TODO 3 * (self._max_caption_length//4
                    temperature=1, #TODO change here
            )

            for i in range(out.size()[0]):
                first_rat = out[i].tolist() #TODO changed from out[0, len(context_tokens):].tolist()
                text = self.gpt2_tokenizer.decode(first_rat, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                # text = text[: text.find(self.gpt2_tokenizer.stop_token)]

                rationale_text = self.gpt2_tokenizer.decode(rationale_text_label[i].tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=True)
                # rationale_text = rationale_text[: rationale_text.find(self.gpt2_tokenizer)]

                q_id_f = (q_id[i] - 1000000).item()
                pred_ans_f = pred_ans[i].item()
                logger.info("[Img ID: {}] Predicted Ans: {} \t| Gold rationale: {} | Generated rationale: {}".format(q_id_f, pred_ans_f, rationale_text, text))

                for rat_ids, gen_ids in zip(rationale_text_label.tolist(), out.tolist()):
                    rat_dec = self.gpt2_tokenizer.decode(rat_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                    gen_dec = self.gpt2_tokenizer.decode(gen_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                    references.append(rat_dec)
                    hypotheses.append(gen_dec)

            bleu_score=sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
            to_return = to_return + (bleu_score,)

        return to_return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--use_chunk", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='BertAdam', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze", default = -1, type=int,
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--vision_scratch", action="store_true", help="whether pre-trained the image or not."
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler", default='mannul', type=str, help="whether use learning rate scheduler."
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--compact", action="store_true", help="whether use compact vilbert model."
    )
    parser.add_argument(
        "--debug", action="store_true", help="whether in debug mode."
    )
    parser.add_argument(
        "--tensorboard_dir",
        default="tensorboard_log",
        type=str,
        help="The output directory where tensorboard log will be written.",
    )
    parser.add_argument(
        "--batch_size",
        default=-1,
        type=int,
        help="Custom Batch size for task.",
    )
    parser.add_argument(
        "--data_root",
        default="",
        type=str,
        help="The data root of the task.",
    )
    args = parser.parse_args()
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    elif args.compact:
        from vilbert.vilbert_compact import BertConfig
        from vilbert.vilbert_compact import VILBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)
        task_lr.append(task_cfg[task]['lr'])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name:
        prefix = '-' + args.save_name
    else:
        prefix = ''

    timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)
    logPath = os.path.join(args.tensorboard_dir, timeStamp)

    # removes everything in that directory
    if os.path.isdir(logPath):
        logger.error('Tensorboard Log path exists. Overwriting.')

    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, 'command.txt'), 'w') as f:
            print(args, file=f)  # Python 3.x
            print('\n', file=f)
            print(config, file=f)

    if args.batch_size != -1:
        for i, task_id in enumerate(args.tasks.split('-')):
            task = 'TASK' + task_id
            task_cfg[task]['batch_size'] = args.batch_size

    if args.data_root != "":
        for i, task_id in enumerate(args.tasks.split('-')):
            data_root = args.data_root
            task = 'TASK' + task_id
            task_cfg[task]['dataroot'] = data_root
            task_cfg[task]['features_h5path1'] = os.path.join(data_root ,task_cfg[task]['features_h5path1'].split('/')[-1])
            task_cfg[task]['features_h5path2'] = os.path.join(data_root ,task_cfg[task]['features_h5path2'].split('/')[-1])
            task_cfg[task]['train_annotations_jsonpath'] = os.path.join(data_root ,task_cfg[task]['train_annotations_jsonpath'].split('/')[-1])
            task_cfg[task]['val_annotations_jsonpath'] = os.path.join(data_root ,task_cfg[task]['val_annotations_jsonpath'].split('/')[-1])

    # Done it for VCR Dataset only, need to put this train_100.jsonl for other datasets
    if args.debug:
        for i, task_id in enumerate(args.tasks.split('-')):
            task = 'TASK' + task_id
            task_cfg[task]['train_annotations_jsonpath'] = '/'.join(task_cfg[task]['train_annotations_jsonpath'].split('/')[:-1] + ['train_100.jsonl'])
            task_cfg[task]['val_annotations_jsonpath'] = '/'.join(task_cfg[task]['val_annotations_jsonpath'].split('/')[:-1] + ['val_100.jsonl'])
            task_cfg[task]['batch_size'] = 90

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    # Have added args.debug to only VCR Datasets (vcr_dataset.py) will need to add it to other dataset too.
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, \
            task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, gpt2_tokenizer, args.tasks.split('-'), args.debug)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    tbLogger = utils.tbLogger(logPath, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps)

    # if n_gpu > 0:
        # torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = max(task_num_iters.values()) * args.num_train_epochs // args.gradient_accumulation_steps
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.baseline:
        vil_model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        vil_model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels, default_gpu=default_gpu
            )

    model = ViLBertGPT2(vil_model, gpt2_tokenizer, gpt2_embed_dim=768, config=config)
    PATH='save/trained_models/vilbert_gpt2_loss_3_1.bin'
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    task_losses = LoadLosses(args, task_cfg, args.tasks.split('-'))
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    model.eval()
    # when run evaluate, we run each task sequentially.
    epochId=0
    for task_id in task_ids:
        num_batch_10 = int(0.1*len(task_dataloader_val[task_id]))
        if args.debug:
            num_batch_10=1
        for i, batch in enumerate(task_dataloader_val[task_id]):
            # generate
            if i%num_batch_10==0:
                generate = True
                loss_vl, gpt2_loss, score, batch_size, bleu_score = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses, generate=generate)
            else:
                generate = False
                loss_vl, gpt2_loss, score, batch_size = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses, generate=generate)

            loss = loss_vl + gpt2_loss
            tbLogger.step_val(epochId, float(loss), float(loss_vl), float(gpt2_loss), float(score), bleu_score, task_id, batch_size, 'val')

            if default_gpu:
                sys.stdout.write('%d/%d\r' % (i, len(task_dataloader_val[task_id])))
                sys.stdout.flush()

            epochId += 1

    ave_score = tbLogger.showLossVal()
    tbLogger.txt_close()

if __name__ == "__main__":

    main()
