import matplotlib

matplotlib.use('Agg')
from utils.data_reader import Personas_CVAE
from model.transformer import Transformer
from model.common_layer import NoamOpt, evaluate
from utils import config
from model.CVAE.util.config import Model_Config
import torch
import torch.nn as nn
from model.CVAE.Optim import Optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import numpy as np
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tensorboardX import SummaryWriter
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from model.CVAE.model import Model


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def make_infinite_list(personas):
    while True:
        print("New epoch(shuffle all personas)")
        shuffle(personas)
        for x in personas:
            yield x


def do_learning(model, train_iter, iterations):
    p, l = [], []
    for i in range(iterations):
        # print(train_iter.__next__())
        loss, ppl, _ = model.train_one_batch(train_iter.__next__())
        l.append(loss)
        p.append(ppl)
    return loss


def do_learning_early_stop(model, train_iter, val_iter, iterations, strict=1):
    # b_loss, b_ppl = do_evaluation(model, val_iter)
    b_loss, b_ppl = 100000, 100000
    best = deepcopy(model.state_dict())
    cnt = 0
    idx = 0
    for _, _ in enumerate(range(iterations)):
        train_l, train_p = [], []
        for d in train_iter:
            t_loss, t_ppl, _ = model.train_one_batch(d)
            train_l.append(t_loss)
            train_p.append(t_ppl)

        n_loss, n_ppl = do_evaluation(model, val_iter)
        ## early stopping
        if (n_ppl <= b_ppl):
            b_ppl = n_ppl
            b_loss = n_loss
            cnt = 0
            idx += 1
            best = deepcopy(model.state_dict())  ## save best weights
        else:
            cnt += 1
        if (cnt > strict): break

    ## load the best model 
    model.load_state_dict({name: best[name] for name in best})

    return (np.mean(train_l), np.mean(train_p), b_loss, b_ppl), idx


def do_learning_fix_step(model, train_iter, val_iter, iterations, test=False):
    val_p = []
    val_p_list = []
    val_loss = 0
    for _, _ in enumerate(range(iterations)):

        for d in train_iter:
            t_loss, t_ppl, _ = model.train_one_batch(d)
        if test:
            _, test_ppl = do_evaluation(model, val_iter)
            val_p_list.append(test_ppl)
    # weight = deepcopy(model.state.dict())

    if test:
        return val_p_list
    else:
        for d in val_iter:
            _, t_ppl, t_loss = model.train_one_batch(d, train=False)
            val_loss += t_loss
            val_p.append(t_ppl)
        return val_loss, np.mean(val_p)


def do_evaluation(model, test_iter):
    p, l = [], []
    for batch in test_iter:
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
    return np.mean(l), np.mean(p)


# =================================main=================================

# Model config for CVAE-dialog
model_config = Model_Config()

# p = Personas()
p = Personas_CVAE()
writer = SummaryWriter(log_dir=config.save_path)
# Build model, optimizer, and set states
if not (config.load_frompretrain == 'None'):
    # meta_net = Transformer(p.vocab, model_file_path=config.load_frompretrain, is_eval=False)
    meta_net = Model(model_config, p.vocab)
else:
    # meta_net = Transformer(p.vocab)
    meta_net = Model(model_config, p.vocab)
    meta_net.print_parameters()  # 输出模型参数个数
    if config.USE_CUDA:
        meta_net.to('cuda')
if config.meta_optimizer == 'sgd':
    meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=config.meta_lr)
elif config.meta_optimizer == 'adam':
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=config.meta_lr)
elif config.meta_optimizer == 'noam':
    meta_optimizer = NoamOpt(config.hidden_dim, 1, 4000,
                             torch.optim.Adam(meta_net.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
else:
    raise ValueError

# CVAE-Optimizer
model_optim = Optim(model_config.method, model_config.lr, model_config.lr_decay, model_config.weight_decay,
                    model_config.max_grad_norm)
model_optim.set_parameters(meta_net.parameters())  # 给优化器设置参数
meta_net.optim = model_optim

meta_batch_size = config.meta_batch_size
tasks = p.get_personas('train')
# tasks_loader = {t: p.get_data_loader(persona=t,batch_size=config.batch_size, split='train') for t in tasks}
tasks_iter = make_infinite_list(tasks)

# meta early stop
patience = 50
if config.fix_dialnum_train:
    patience = 100
# best_loss = 10000000
best_loss = 100
stop_count = 0
# Main loop
for meta_iteration in range(config.epochs):

    ## save original weights to make the update
    weights_original = deepcopy(meta_net.state_dict())
    train_loss_before = []
    train_loss_meta = []
    # loss accumulate from a batch of tasks
    batch_loss = 0
    batch_loss_before = 0
    for _ in range(meta_batch_size):
        # Get task
        if config.fix_dialnum_train:
            train_iter, val_iter = p.get_balanced_loader(persona=tasks_iter.__next__(), batch_size=config.batch_size,
                                                         split='train')
        else:
            train_iter, val_iter = p.get_data_loader(persona=tasks_iter.__next__(), batch_size=config.batch_size,
                                                     split='train')
        # before first update
        v_loss, v_ppl = do_evaluation(meta_net, val_iter)
        train_loss_before.append(math.exp(v_loss))
        batch_loss_before += v_loss
        # Update fast nets   
        val_loss, v_ppl = do_learning_fix_step(meta_net, train_iter, val_iter, iterations=config.meta_iteration)
        train_loss_meta.append(math.exp(val_loss.item()))
        batch_loss += val_loss
        # log

        # reset 
        meta_net.load_state_dict({name: weights_original[name] for name in weights_original})

    writer.add_scalars('loss_before', {'train_loss_before': np.mean(train_loss_before)}, meta_iteration)
    writer.add_scalars('loss_meta', {'train_loss_meta': np.mean(train_loss_meta)}, meta_iteration)
    # print(' loss_before: {}'.format(np.mean(train_loss_before)))
    # print(' loss_meta: {}'.format(np.mean(np.mean(train_loss_meta))))

    meta_net.optim.update_lr(meta_iteration)  # 每个epoch更新学习率

    # meta Update
    if (config.meta_optimizer == 'noam'):
        meta_optimizer.optimizer.zero_grad()
    else:
        meta_optimizer.zero_grad()

    batch_loss_before /= meta_batch_size
    batch_loss /= meta_batch_size
    batch_loss.backward()
    print('loss_before: {:.3f}'.format(batch_loss_before))
    print('loss_after: {:.3f}'.format(batch_loss))


    # clip gradient
    nn.utils.clip_grad_norm_(meta_net.parameters(), config.max_grad_norm)
    meta_optimizer.step()

    ## Meta-Evaluation
    if meta_iteration % 10 == 0:
        print('Validation')
        print('    Meta_iteration:', meta_iteration)
        val_loss_before = []
        val_loss_meta = []
        val_batch_loss_before = 0
        val_batch_loss_after = 0
        weights_original = deepcopy(meta_net.state_dict())
        for idx, per in enumerate(p.get_personas('valid')):
            # num_of_dialog = p.get_num_of_dialog(persona=per, split='valid')
            # for dial_i in range(num_of_dialog):
            if config.fix_dialnum_train:
                train_iter, val_iter = p.get_balanced_loader(persona=per, batch_size=config.batch_size, split='valid',
                                                             fold=0)

            else:
                train_iter, val_iter = p.get_data_loader(persona=per, batch_size=config.batch_size, split='valid',
                                                         fold=0)
            # zero shot result
            loss, ppl = do_evaluation(meta_net, val_iter)
            val_loss_before.append(math.exp(loss))
            val_batch_loss_before += loss
            # mate tuning
            val_loss, val_ppl = do_learning_fix_step(meta_net, train_iter, val_iter, iterations=config.meta_iteration)
            val_loss_meta.append(math.exp(val_loss.item()))
            val_batch_loss_after += val_loss
            # updated result

            meta_net.load_state_dict({name: weights_original[name] for name in weights_original})

        writer.add_scalars('loss_before', {'val_loss_before': np.mean(val_loss_before)}, meta_iteration)
        writer.add_scalars('loss_meta', {'val_loss_meta': np.mean(val_loss_meta)}, meta_iteration)

        val_batch_loss_before /= 100
        val_batch_loss_after /= 100
        print('    val_loss: {:.3f} ===> {:.3f}'.format(val_batch_loss_before, val_batch_loss_after))


        # check early stop
        if np.mean(val_loss_meta) < best_loss:
            best_loss = np.mean(val_loss_meta)
            stop_count = 0
            meta_net.save_model(best_loss, meta_iteration, 0)
        else:
            stop_count += 1
            if stop_count > patience:
                break
