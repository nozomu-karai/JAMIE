#!/usr/bin/env python
# coding: utf-8
import warnings
import os
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import *

import clinical_eval
from clinical_eval import MhsEvaluator
import utils
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description='PRISM joint recognizer')

    parser.add_argument("--train_file",
                        default="data/2021Q1/mr150/doc_conll/cv0_train.conll",
                        type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file",
                        default="data/2021Q1/mr150/doc_conll/cv0_dev.conll",
                        type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file",
                        default="data/2021Q1/mr150/doc_conll/cv0_test.conll",
                        type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pretrained_model",
                        default="/home/feicheng/Tools/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--saved_model", default='checkpoints/tmp/joint_mr_doc', type=str,
                        help="save/load model dir")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="tokenizer: do_lower_case")

    parser.add_argument("--test_output", default='tmp/mr_rev.test.conll', type=str,
                        help="test output filename")

    parser.add_argument("--dev_output", default='tmp/mr_rev.dev.conll', type=str,
                        help="dev output filename")

    parser.add_argument("--test_dir", default="tmp/", type=str,
                        help="test dir, multihead conll format.")

    parser.add_argument("--pred_dir", default="tmp/", type=str,
                        help="prediction dir, multihead conll format.")

    parser.add_argument("--batch_test",
                        action='store_true',
                        help="test batch files")

    parser.add_argument("--batch_size", default=4, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=30, type=int,
                        help="fine-tuning epoch number")

    parser.add_argument("--embed_size", default='[32, 32, 832]', type=str,
                        help="ner, mod, rel embedding size")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--freeze_after_epoch", default=50, type=int,
                        help="freeze encoder after N epochs")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--enc_lr", default=2e-5, type=float,
                        help="learning rate")

    parser.add_argument("--dec_lr", default=1e-2, type=float,
                        help="learning rate")

    parser.add_argument("--other_lr", default=1e-3, type=float,
                        help="learning rate")

    parser.add_argument("--reduction", default='token_mean', type=str,
                        help="loss reduction: `token_mean` or `sum`")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_portion", default=2, type=int,
                        help="save best model given a portion of steps")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    parser.add_argument("--warmup_epoch", default=2, type=float,
                        help="warmup epoch")

    parser.add_argument("--scheduled_lr",
                        action='store_true',
                        help="learning rate schedule")

    parser.add_argument("--epoch_eval",
                        action='store_true',
                        help="eval each epoch")

    parser.add_argument("--fp16",
                        action='store_true',
                        help="fp16")

    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    print("----------args---------------")
    print(args) 

    bert_max_len = 512

    bio_emb_size, mod_emb_size, rel_emb_size = eval(args.embed_size)

    if args.do_train:

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
        tokenizer.add_tokens(['[JASP]'])

        train_comments, train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
            args.train_file,
            down_neg=0.0
        )

if __name__ == '__main__':
    main()