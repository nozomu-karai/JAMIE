#!/usr/bin/env python
# coding: utf-8
import warnings
import random
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import *
from tensorboardX import SummaryWriter

import argparse
from model import *
from clinical_eval import MhsEvaluator
from collections import defaultdict
from data_objects import bio_to_spans
warnings.filterwarnings("ignore")


def generate_batch_pair_mask_t(doc_pair_mask, batch_sent_ids, cls_max_len):

    batch_tail_mask = [[pair_mask[0] for pair_mask in doc_pair_mask[sent_id]] for sent_id in batch_sent_ids]
    batch_head_mask = [[pair_mask[1] for pair_mask in doc_pair_mask[sent_id]] for sent_id in batch_sent_ids]
    pair_max_num = max([len(sent_pair) for sent_pair in batch_tail_mask])
    padded_doc_tail_mask_ix_t = torch.tensor(
        padding_3d(
            batch_tail_mask,
            cls_max_len,
            pair_max_num
        )
    )
    padded_doc_head_mask_ix_t = torch.tensor(
        padding_3d(
            batch_head_mask,
            cls_max_len,
            pair_max_num
        )
    )
    padded_doc_pair_mask_ix_t = torch.cat((padded_doc_tail_mask_ix_t, padded_doc_head_mask_ix_t), dim=-1)
    return padded_doc_pair_mask_ix_t


def generate_batch_pair_tag_t(doc_pair_tag, batch_sent_ids, ne2ix):
    batch_tail_tag = [[pair_tag[0] for pair_tag in doc_pair_tag[sent_id]] for sent_id in batch_sent_ids]
    batch_head_tag = [[pair_tag[1] for pair_tag in doc_pair_tag[sent_id]] for sent_id in batch_sent_ids]
    pair_max_num = max([len(sent_tag) for sent_tag in batch_tail_tag])
    padded_doc_tail_tag_t = torch.tensor(
        [padding_1d(
            [ne2ix[tag] if tag in ne2ix else ne2ix['O'] for tag in sent_tail],
            pair_max_num,
            pad_tok=0
        ) for sent_tail in batch_tail_tag]
    )
    padded_doc_head_tag_t = torch.tensor(
        [padding_1d(
            [ne2ix[tag] if tag in ne2ix else ne2ix['O'] for tag in sent_head],
            pair_max_num,
            pad_tok=0
        ) for sent_head in batch_head_tag]
    )
    return padded_doc_tail_tag_t, padded_doc_head_tag_t


def generate_batch_rel_t(doc_rel, batch_sent_ids, rel2ix, neg_ratio):
    batch_rel = [doc_rel[sent_id] for sent_id in batch_sent_ids]
    rel_max_num = max([len(sent_rel) for sent_rel in batch_rel])
    padded_doc_rel_ix_t = torch.tensor(
        [padding_1d(
            [-100 if (rel == 'N' and random.random() > neg_ratio) else rel2ix[rel] for rel in sent_rel],
            rel_max_num,
            pad_tok=-100
        ) for sent_rel in batch_rel]
    )
    return padded_doc_rel_ix_t


def output_rel(
        trained_model,
        eval_dataloader, eval_comment, eval_tok, eval_ner, eval_mod, eval_pair_mask, eval_pair_tag,
        rel2ix, cls_max_len, rel_outfile, device, test_mode=False
):
    ix2rel = {v: k for k, v in rel2ix.items()}
    trained_model.eval()
    with torch.no_grad(), open(rel_outfile, 'w') as fo:
        for dev_batch in tqdm(eval_dataloader, desc="Testing", disable=not test_mode):
            b_e_toks, b_e_attn_mask, b_e_sent_mask, b_e_ner, b_e_ner_mask, b_e_mod = tuple(
                t.to(device) for t in dev_batch[1:]
            )
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_e_pair_mask = generate_batch_pair_mask_t(eval_pair_mask, b_sent_ids, cls_max_len).to(device)
            if len(b_e_pair_mask.shape) > 2:
                b, e, l = b_e_pair_mask.shape
                b_e_pair_tail, b_e_pair_head = generate_batch_pair_tag_t(eval_pair_tag, b_sent_ids, ne2ix)
                pred_logit = trained_model(
                    b_e_toks, b_e_pair_mask.float(),
                    b_e_pair_tail.to(device), b_e_pair_head.to(device), attention_mask=b_e_attn_mask.bool())
                pred_tag_ix = pred_logit.argmax(-1).view(-1).cpu()  # flatten batch x entity
                tag_mask = torch.tensor([True if m != [0] * l else False for m in b_e_pair_mask.view(-1, l).tolist()])
                pred_tag = pred_tag_ix.masked_select(tag_mask).tolist()

            for sid in b_sent_ids:
                w_tok, aligned_ids = utils.sbwtok2tok_alignment(eval_tok[sid])
                w_ner = utils.sbwner2ner(eval_ner[sid], aligned_ids)
                w_mod = utils.sbwmod2mod(eval_mod[sid], aligned_ids)
                w_tok = w_tok[1:-1]
                w_ner = w_ner[1:-1]
                w_mod = w_mod[1:-1]
                assert len(w_tok) == len(w_ner) == len(w_mod)

                if len(b_e_pair_mask.shape) > 2:
                    sent_spans = bio_to_spans(w_ner)
                    last_tid2head = defaultdict(list)
                    last_tid2rel = defaultdict(list)
                    for t_index, (t_ner, t_start, t_end) in enumerate(sent_spans):
                        for h_index, (h_ner, h_start, h_end) in enumerate(sent_spans):
                            tmp_rel = ix2rel[pred_tag.pop(0)]
                            # if h_index != t_index:
                            if tmp_rel != 'N':
                                last_tid2head[t_end - 1].append(h_end - 1)
                                last_tid2rel[t_end - 1].append(tmp_rel)
                            if not last_tid2head[t_end - 1] and not last_tid2rel[t_end - 1]:
                                last_tid2head[t_end - 1] = [t_end - 1]
                                last_tid2rel[t_end - 1] = ['N']

                    fo.write(f'{eval_comment[sid]}\n')
                    for index, (tok, ner, mod) in enumerate(zip(w_tok, w_ner, w_mod)):
                        head_col = last_tid2head[index] if index in last_tid2head else f'[{index}]'
                        rel_col = last_tid2rel[index] if index in last_tid2rel else "['N']"
                        fo.write(f"{index}\t{tok}\t{ner}\t{mod}\t{rel_col}\t{head_col}\n")
                else:
                    fo.write(f'{eval_comment[sid]}\n')
                    for index, (tok, ner, mod) in enumerate(zip(w_tok, w_ner, w_mod)):
                        fo.write(f"{index}\t{tok}\t{ner}\t{mod}\t['N']\t[{index}]\n")
            # if len(b_e_pair_mask.shape) > 2:
            #     print(f'left pred_tag: {len(pred_tag)}')

""" 
python input arguments 
"""
parser = argparse.ArgumentParser(description='Clinical IE pipeline relation extraction')

parser.add_argument("--pretrained_model",
                    default="/home/feicheng/Tools/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                    type=str,
                    help="pre-trained model dir")

parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="tokenizer: do_lower_case")

parser.add_argument("--saved_model", default='checkpoints/tmp/pipeline/ncc/rel', type=str,
                    help="save/load model dir")

parser.add_argument("--train_file", default="data/2020Q2/ncc20200601_rev/sent_conll/cv0_train.conll", type=str,
                    help="train file, multihead conll format.")

parser.add_argument("--dev_file", default="data/2020Q2/ncc20200601_rev/sent_conll/cv0_dev.conll", type=str,
                    help="dev file, multihead conll format.")

parser.add_argument("--test_file", default="data/2020Q2/ncc20200601_rev/sent_conll/cv0_test.conll", type=str,
                    help="test file, multihead conll format.")

parser.add_argument("--batch_size", default=16, type=int,
                    help="BATCH SIZE")

parser.add_argument("--num_epoch", default=10, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--enc_lr", default=2e-5, type=float,
                    help="encoder lr")

parser.add_argument("--dec_lr", default=1e-3, type=float,
                    help="decoder layer lr")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--test_output", default='tmp/ncc.test.rel', type=str,
                    help="test output filename")

parser.add_argument("--dev_output", default='tmp/ncc.dev.rel', type=str,
                    help="dev output filename")

parser.add_argument("--epoch_start_eval", default=3, type=int,
                    help="epoch num starting eval with validation data")

parser.add_argument("--later_eval",
                    action='store_true',
                    help="Whether eval model every epoch.")

parser.add_argument("--save_best", action='store', type=str, default='f1',
                    help="save the best model, given dev scores (f1 or loss)")

parser.add_argument("--logging_interval", default=3, type=int,
                    help="save best model given a portion of steps")

parser.add_argument("--warmup_epoch", default=3, type=float,
                    help="warmup epoch")

parser.add_argument("--neg_ratio", default=1.0, type=float,
                    help="negative sampling ratio")

parser.add_argument("--fp16",
                    action='store_true',
                    help="fp16")

parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument("--scheduled_lr",
                    action='store_true',
                    help="learning rate schedule")


args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', args.device)
args.n_gpu = torch.cuda.device_count()

if args.do_train:
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case, do_basic_tokenize=False)
    tokenizer.add_tokens(['[JASP]'])

    """ Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
    train_comments, train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
        args.train_file,
        down_neg=0.0
    )
    max_word_len = max([len(sent_tok) for sent_tok in train_toks])
    max_len_train = utils.max_sents_len(train_toks, tokenizer)
    print(bio2ix)
    print(ne2ix)
    print(mod2ix)
    print(rel2ix)
    print()
    print(f'max training tok len: {max_len_train}, max training word len: {max_word_len}')
    print()

    dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
        args.dev_file,
        down_neg=0.0
    )
    max_len_dev = utils.max_sents_len(dev_toks, tokenizer)
    print('max dev sent len:', )
    print()

    max_len = max(max_len_train, max_len_dev)
    cls_max_len = max_len + 2
    print(f"max seq len: {max_len}, max seq len with [CLS] and [SEP]: {cls_max_len}")

    example_id = 7

    print(f"Random example: id {example_id}, len: {len(train_toks[example_id])}")
    for tok_id in range(len(train_toks[example_id])):
        print(f"{tok_id}\t{train_toks[example_id][tok_id]}\t{train_ners[example_id][tok_id]}")
    print(train_rels[example_id])
    print()

    """ 
    - Generate train/test tensors including (token_ids, mask_ids, label_ids) 
    - wrap them into dataloader for mini-batch cutting
    """
    train_dataset, train_comment, train_tok, train_ner, train_mod, \
    train_pair_mask, train_pair_tag, train_rel, train_rel_tup, train_spo = utils.extract_pipeline_data_from_mhs_conll(
        train_comments, train_toks, train_ners, train_mods, train_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

    dev_dataset, dev_comment, dev_tok, dev_ner, dev_mod, \
    dev_pair_mask, dev_pair_tag, dev_rel, dev_rel_tup, dev_spo = utils.extract_pipeline_data_from_mhs_conll(
        dev_comments, dev_toks, dev_ners, dev_mods, dev_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    """
    Model
    """
    model = PipelineRelation(args.pretrained_model, len(ne2ix), len(rel2ix))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # specify different lr
    param_optimizer = list(model.named_parameters())
    encoder_name_list = ['encoder']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in encoder_name_list)], 'lr': args.dec_lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in encoder_name_list)], 'lr': args.enc_lr}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
    )

    # PyTorch scheduler
    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    logging_steps = math.ceil(num_epoch_steps / args.logging_interval)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_epoch_steps * args.warmup_epoch,
        num_training_steps=num_training_steps
    )

    # support fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    """
    Start training loop
    """
    tb_writer = SummaryWriter()
    best_dev_f1 = (float('-inf'), 0, 0)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for epoch in range(1, args.num_epoch + 1):

        epoch_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(train_dataloader))
        for step, batch in enumerate(epoch_iterator):

            model.train()

            b_toks, b_attn_mask, b_sent_mask, b_ner, b_ner_mask, b_mod = tuple(
                t.to(args.device) for t in batch[1:]
            )

            b_sent_ids = batch[0].tolist()

            b_pair_mask = generate_batch_pair_mask_t(train_pair_mask, b_sent_ids, cls_max_len).to(args.device)
            b_pair_tail, b_pair_head = generate_batch_pair_tag_t(train_pair_tag, b_sent_ids, ne2ix)
            b_rel = generate_batch_rel_t(train_rel, b_sent_ids, rel2ix, args.neg_ratio).to(args.device)
            if len(b_pair_mask.shape) < 3:
                continue
            # BERT loss, logits: (batch_size, seq_len, tag_num)
            loss = model(b_toks, b_pair_mask.float(), b_pair_tail.to(args.device), b_pair_head.to(args.device),
                         attention_mask=b_attn_mask.bool(), labels=b_rel)

            epoch_loss += loss.item()
            tr_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                f"L_REL: {epoch_loss / (step + 1):.6f} | epoch: {epoch}/{args.num_epoch}:"
            )

            if epoch > args.epoch_start_eval:
                if (step + 1) % logging_steps == 0:

                    '''logging tensorboardx: lr, loss'''
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                    '''dev eval'''
                    output_rel(
                        model, dev_dataloader,
                        dev_comments, dev_tok, dev_ner, dev_mod, dev_pair_mask, dev_pair_tag,
                        rel2ix, cls_max_len, args.dev_output, args.device
                    )
                    dev_evaluator = MhsEvaluator(args.dev_file, args.dev_output)
                    dev_f1 = (dev_evaluator.eval_rel(print_level=0), epoch, step)

                    '''save best model'''
                    if best_dev_f1[0] < dev_f1[0]:
                        print(
                            f" -> Previous best dev f1 {best_dev_f1[0]:.6f}; "
                            f"epoch {best_dev_f1[1]:d} / step {best_dev_f1[2]:d} \n "
                            f">> Current f1 {dev_f1[0]:.6f}; best model saved '{args.saved_model}'"
                        )
                        best_dev_f1 = dev_f1

                        """ save the best model """
                        if not os.path.exists(args.saved_model):
                            os.makedirs(args.saved_model)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(args.saved_model, 'model.pt'))
                        tokenizer.save_pretrained(args.saved_model)
                        with open(os.path.join(args.saved_model, 'ner2ix.json'), 'w') as fp:
                            json.dump(bio2ix, fp)
                        with open(os.path.join(args.saved_model, 'ne2ix.json'), 'w') as fp:
                            json.dump(ne2ix, fp)
                        with open(os.path.join(args.saved_model, 'mod2ix.json'), 'w') as fp:
                            json.dump(mod2ix, fp)
                        with open(os.path.join(args.saved_model, 'rel2ix.json'), 'w') as fp:
                            json.dump(rel2ix, fp)
                    if num_training_steps == step + 1:
                        dev_evaluator.eval_rel(print_level=1)
        if epoch > args.epoch_start_eval:
            dev_evaluator = MhsEvaluator(args.dev_file, args.dev_output)
            dev_evaluator.eval_rel(print_level=1)
    print(f"Best dev f1 {best_dev_f1[0]:.6f}; epoch {best_dev_f1[1]:d} / step {best_dev_f1[2]:d}\n")
    model.load_state_dict(torch.load(os.path.join(args.saved_model, 'model.pt')))
    torch.save(model, os.path.join(args.saved_model, 'model.pt'))
    output_rel(
        model, dev_dataloader,
        dev_comments, dev_tok, dev_ner, dev_mod, dev_pair_mask, dev_pair_tag,
        rel2ix, cls_max_len, args.dev_output, args.device
    )
    tb_writer.close()
    dev_evaluator = MhsEvaluator(args.dev_file, args.dev_output)
    dev_evaluator.eval_rel(print_level=2)
else:
    """ load the new tokenizer"""
    print("test_mode:", args.saved_model)
    tokenizer = BertTokenizer.from_pretrained(
        args.saved_model,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=False
    )
    with open(os.path.join(args.saved_model, 'ner2ix.json')) as json_fi:
        bio2ix = json.load(json_fi)
    with open(os.path.join(args.saved_model, 'ne2ix.json')) as json_fi:
        ne2ix = json.load(json_fi)
    with open(os.path.join(args.saved_model, 'mod2ix.json')) as json_fi:
        mod2ix = json.load(json_fi)
    with open(os.path.join(args.saved_model, 'rel2ix.json')) as json_fi:
        rel2ix = json.load(json_fi)

    """ load test data """
    test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
        args.test_file,
        down_neg=0.0)
    print(f"max sent len: {utils.max_sents_len(test_toks, tokenizer)}")
    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
    print()

    max_len = utils.max_sents_len(test_toks, tokenizer)
    cls_max_len = max_len + 2

    test_dataset, test_comment, test_tok, test_ner, test_mod, test_pair_mask, test_pair_tag, test_rel, test_rel_tup, test_spo = utils.extract_pipeline_data_from_mhs_conll(
        test_comments, test_toks, test_ners, test_mods, test_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """ load the saved model"""
    model = torch.load(os.path.join(args.saved_model, 'model.pt'))
    model.to(args.device)

    """ predict test out """
    output_rel(
        model, test_dataloader,
        test_comments, test_tok, test_ner, test_mod, test_pair_mask, test_pair_tag,
        rel2ix, cls_max_len, args.test_output, args.device, test_mode=True)
    test_evaluator = MhsEvaluator(args.test_file, args.test_output)
    test_evaluator.eval_rel(print_level=1)
