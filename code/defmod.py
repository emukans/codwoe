import argparse
import itertools
import json
import logging
import os
import pathlib
import sys
from random import random

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models


def get_parser(
    parser=argparse.ArgumentParser(description="run a definition modeling baseline"),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--data_dir", type=pathlib.Path, default=pathlib.Path("data"), help="path to the train file"
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--source_arch",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as source",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / "defmod-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / "defmod-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("defmod"),
        help="where to save predictions",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="base language",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="prefix to save directory and logs",
    )
    return parser


def train(args):
    assert args.data_dir is not None, "Missing dataset for training"

    ## Hyperparams
    INPUT_SIZE = 256
    HIDDEN_SIZE = 768
    N_LAYERS = 2
    EPOCHS = 500
    LEARNING_RATE = 3.0e-5
    BETA1 = 0.9
    BETA2 = 0.999
    WEIGHT_DECAY = 1.0e-2
    MAX_LEN = 30
    BATCH_SIZE = 384
    DROPOUT = 0.1
    PATIENCE = 10
    SPM_MODEL_NAME = f'spm_model_{args.lang}'
    label_smoothing = 0.1

    if args.prefix:
        save_dir = args.save_dir.joinpath(f'{args.prefix}/{args.lang}/{args.source_arch}')
        log_dir = args.summary_logdir.joinpath(f'{args.prefix}-{args.lang}-{args.source_arch}')
    else:
        save_dir = args.save_dir.joinpath(f'{args.lang}/{args.source_arch}')
        log_dir = args.summary_logdir.joinpath(f'{args.lang}-{args.source_arch}')


    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading training data")
    ## make datasets
    train_dataset = data.JSONDataset(args.data_dir.joinpath(f'{args.lang}.train.json'), maxlen=MAX_LEN, spm_model_name=SPM_MODEL_NAME)
    dev_dataset = data.JSONDataset(args.data_dir.joinpath(f'{args.lang}.dev.json'), vocab=train_dataset.vocab, maxlen=MAX_LEN, spm_model_name=SPM_MODEL_NAME)

    os.makedirs(save_dir, exist_ok=True)
    train_dataset.save(save_dir / "train_dataset.pt")
    dev_dataset.save(save_dir / "dev_dataset.pt")

    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.source_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    if args.source_arch == "electra":
        assert dev_dataset.has_electra, "Development dataset contains no vector."
    else:
        assert dev_dataset.has_vecs, "Development dataset contains no vector."

    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset, batch_size=BATCH_SIZE)
    dev_dataloader = data.get_dataloader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    ## make summary writer
    summary_writer = SummaryWriter(log_dir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    logger.debug("Setting up training environment")

    # model = models.LSTMDefmodModel(dev_dataset.vocab, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, maxlen=MAX_LEN, dropout=DROPOUT).to(args.device)
    model = models.DefmodModel(dev_dataset.vocab, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, maxlen=MAX_LEN, dropout=DROPOUT).to(args.device)

    model.train()

    # 3. declare optimizer & criterion
    optimizer = configure_optimizers(BETA1, BETA2, LEARNING_RATE, WEIGHT_DECAY, model)

    xent_criterion = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    if label_smoothing > 0.0:
        smooth_criterion = models.LabelSmoothingCrossEntropy(
            ignore_index=model.padding_idx, epsilon=label_smoothing
        )
    else:
        smooth_criterion = xent_criterion

    vec_tensor_key = f"{args.source_arch}_tensor"
    max_acc = float('-inf')
    strikes = 0
    epochs_range = tqdm.trange(EPOCHS, desc="Epochs")

    # 4. train model
    for epoch in epochs_range:
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for batch in train_dataloader:
            optimizer.zero_grad()
            vec = batch[vec_tensor_key].to(args.device)
            gls = batch["gloss_tensor"].to(args.device)

            real_batch_size = batch[vec_tensor_key].size(0)
            hidden = vec.repeat(N_LAYERS, 1, 1).to(args.device)
            encoder_output = vec.unsqueeze(0).to(args.device)
            sequence = torch.tensor([train_dataset.vocab[data.BOS]] * real_batch_size).unsqueeze(0).to(args.device)
            predicted = torch.zeros(0, real_batch_size, len(dev_dataset.vocab)).to(args.device)

            for i in range(MAX_LEN):
                pred, hidden = model(sequence, hidden, encoder_output, i == 0)
                predicted = torch.cat([predicted, pred.unsqueeze(0)], dim=0)

                sequence = gls[:i + 1, :]

            loss = smooth_criterion(predicted.contiguous().view(-1, predicted.size(-1)), gls.contiguous().view(-1))
            loss.backward()
            # keep track of the train loss for this step
            tokens = gls != model.padding_idx
            acc = (
                ((predicted.argmax(-1) == gls) & tokens).float().sum() / tokens.sum()
            ).item()
            step = next(train_step)
            summary_writer.add_scalar("defmod-train/xent", loss.item(), step)
            summary_writer.add_scalar("defmod-train/acc", acc, step)
            optimizer.step()
            pbar.update(vec.size(0))
        pbar.close()
        # eval loop
        model.eval()
        with torch.no_grad():
            sum_dev_loss = 0.0
            sum_acc = 0
            ntoks = 0
            pbar = tqdm.tqdm(
                desc=f"Eval {epoch}",
                total=len(dev_dataset),
                disable=None,
                leave=False,
            )
            for batch in dev_dataloader:
                vec = batch[vec_tensor_key].to(args.device)
                gls = batch["gloss_tensor"].to(args.device)

                real_batch_size = batch[vec_tensor_key].size(0)
                hidden = vec.repeat(N_LAYERS, 1, 1).to(args.device)
                encoder_output = vec.unsqueeze(0).to(args.device)
                sequence = torch.tensor([train_dataset.vocab[data.BOS]] * real_batch_size).unsqueeze(0).to(args.device)
                predicted = torch.zeros(0, real_batch_size, len(dev_dataset.vocab)).to(args.device)
                for i in range(MAX_LEN):
                    pred, hidden = model(sequence, hidden, encoder_output, i == 0)
                    predicted = torch.cat([predicted, pred.unsqueeze(0)], dim=0)

                    sequence = gls[:i + 1, :]

                sum_dev_loss += F.cross_entropy(
                    predicted.contiguous().view(-1, predicted.size(-1)),
                    gls.contiguous().view(-1),
                    reduction="sum",
                    ignore_index=model.padding_idx,
                ).item()
                tokens = gls != model.padding_idx
                ntoks += tokens.sum().item()
                sum_acc += ((predicted.argmax(-1) == gls) & tokens).sum().item()
                pbar.update(vec.size(0))

            avg_acc = sum_acc / ntoks
            # keep track of the average loss & acc on dev set for this epoch
            summary_writer.add_scalar(
                "defmod-dev/xent", sum_dev_loss / ntoks, epoch
            )
            summary_writer.add_scalar("defmod-dev/acc", avg_acc, epoch)
            pbar.close()

            if avg_acc > max_acc:
                max_acc = avg_acc
                # 5. save result
                model.save(save_dir / "model.pt")
                strikes = 0
            else:
                strikes += 1

        if strikes >= PATIENCE:
            logger.debug("Stopping early.")
            epochs_range.close()
            break
        model.train()


def configure_optimizers(BETA1, BETA2, LEARNING_RATE, WEIGHT_DECAY, model):
    """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM, torch.nn.MultiheadAttention)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if (pn.startswith('bias') or pn.endswith('bias')):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.startswith('weight') or pn.endswith('weight')) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.startswith('weight') or pn.endswith('weight')) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                       % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": WEIGHT_DECAY},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(
        optim_groups,
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


def pred(args):
    assert args.data_dir is not None, "Missing dataset for test"

    MAX_LEN = 50
    N_LAYERS = 4
    SPM_MODEL_NAME = f'spm_model_{args.lang}'

    if args.prefix:
        save_dir = args.save_dir.joinpath(f'{args.prefix}/{args.lang}/{args.source_arch}')
        pred_file = f'predictions/{args.pred_file}-{args.prefix}-{args.lang}-{args.source_arch}.json'
    else:
        save_dir = args.save_dir.joinpath(f'{args.lang}/{args.source_arch}')
        pred_file = f'predictions/{args.pred_file}-{args.lang}-{args.source_arch}.json'

    # 1. retrieve vocab, dataset, model
    # model = models.LSTMDefmodModel.load(save_dir / "model.pt")
    model = models.DefmodModel.load(save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.data_dir.joinpath(f'{args.lang}.trial.complete.json'), vocab=train_vocab, freeze_vocab=True, maxlen=256, spm_model_name=SPM_MODEL_NAME
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1)
    model.eval()
    vec_tensor_key = f"{args.source_arch}_tensor"
    if args.source_arch == "electra":
        assert test_dataset.has_electra, "File is not usable for the task"
    else:
        assert test_dataset.has_vecs, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset), disable=None)
        for batch in test_dataloader:
            vec = batch[vec_tensor_key].to(args.device)
            real_batch_size = batch[vec_tensor_key].size(0)

            hidden = vec.repeat(N_LAYERS, 1, 1).to(args.device)
            encoder_output = vec.unsqueeze(0).to(args.device)
            sequence = torch.tensor([train_vocab[data.BOS]], dtype=int).unsqueeze(0).to(args.device)
            predicted = torch.zeros(0, real_batch_size, len(train_vocab)).to(args.device)

            for i in range(MAX_LEN):
                pred, hidden = model(sequence, hidden, encoder_output, i == 0)
                predicted = torch.cat([predicted, pred.unsqueeze(0)], dim=0)

                # teaching forcing
                sequence = torch.cat([sequence, pred.argmax(-1).unsqueeze(0).detach()], dim=0)
                if pred.argmax(-1).item() == train_vocab[data.EOS]:
                    break

            gloss = test_dataset.decode(predicted.argmax(-1).squeeze())
            if random() < 0.05:
                print(gloss)
            predictions.append({"id": batch["id"][0], "gloss": gloss})

            pbar.update(vec.size(0))
        pbar.close()
    # 3. dump predictions

    with open(pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    if args.do_train:
        logger.debug("Performing defmod training")
        train(args)
    if args.do_pred:
        logger.debug("Performing defmod prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
