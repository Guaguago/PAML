import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.CVAE.Embedding import Embedding
from model.CVAE.Encoder import Encoder
from model.CVAE.PriorNet import PriorNet
from model.CVAE.RecognizeNet import RecognizeNet
from model.CVAE.Decoder import Decoder
from model.CVAE.PrepareState import PrepareState
from utils import config
from tqdm import tqdm
from utils.load_bert import bert_model
from utils.metric import moses_multi_bleu
import pprint
from model.CVAE.Optim import Optim

pp = pprint.PrettyPrinter(indent=1)


def print_all(dial, ref, hyp_b, max_print):
    for i in range(len(ref)):
        print(pp.pformat(dial[i]))
        print("Beam: {}".format(hyp_b[i]))
        print("Ref:{}".format(ref[i]))
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        if (i > max_print): break


class Model(nn.Module):
    def __init__(self, model_config, vocab, model_file_path=None, is_eval=False, load_optim=False):
        super(Model, self).__init__()
        self.model_config = model_config
        self.vocab = vocab
        self.model_dir = config.save_path
        # 定义嵌入层
        self.embedding = Embedding(vocab,  # 词汇表大小
                                   model_config.embedding_size,  # 嵌入层维度
                                   model_config.pad_id,  # pad_id
                                   model_config.dropout)

        # post编码器
        self.post_encoder = Encoder(model_config.post_encoder_cell_type,  # rnn类型
                                    model_config.embedding_size,  # 输入维度
                                    model_config.post_encoder_output_size,  # 输出维度
                                    model_config.post_encoder_num_layers,  # rnn层数
                                    model_config.post_encoder_bidirectional,  # 是否双向
                                    model_config.dropout)  # dropout概率

        # response编码器
        self.response_encoder = Encoder(model_config.response_encoder_cell_type,
                                        model_config.embedding_size,  # 输入维度
                                        model_config.response_encoder_output_size,  # 输出维度
                                        model_config.response_encoder_num_layers,  # rnn层数
                                        model_config.response_encoder_bidirectional,  # 是否双向
                                        model_config.dropout)  # dropout概率

        # 先验网络
        self.prior_net = PriorNet(model_config.post_encoder_output_size,  # post输入维度
                                  model_config.latent_size,  # 潜变量维度
                                  model_config.dims_prior)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(model_config.post_encoder_output_size,  # post输入维度
                                          model_config.response_encoder_output_size,  # response输入维度
                                          model_config.latent_size,  # 潜变量维度
                                          model_config.dims_recognize)  # 隐藏层维度

        # 初始化解码器状态
        self.prepare_state = PrepareState(model_config.post_encoder_output_size + model_config.latent_size,
                                          model_config.decoder_cell_type,
                                          model_config.decoder_output_size,
                                          model_config.decoder_num_layers)

        # 解码器
        self.decoder = Decoder(model_config.decoder_cell_type,  # rnn类型
                               model_config.embedding_size,  # 输入维度
                               model_config.decoder_output_size,  # 输出维度
                               model_config.decoder_num_layers,  # rnn层数
                               model_config.dropout)  # dropout概率

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(model_config.decoder_output_size, vocab.n_words),
            nn.Softmax(-1)
        )

        self.bert = bert_model()

        self.optim = Optim(model_config.method, model_config.lr, model_config.lr_decay, model_config.weight_decay,
                           model_config.max_grad_norm)
        self.optim.set_parameters(self.parameters())  # 给优化器设置参数

        self.global_step = 0
        if model_file_path is not None:
            self.load_model(model_file_path)

    def forward(self, inputs, inference=False, max_len=60, gpu=True):
        if not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            len_decoder = id_responses.size(1) - 1

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = self.embedding(id_responses)  # [batch, seq, embed_size]
            # state: [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]
            # 重参数化
            z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]

            # 解码器的输入为回复去掉end_id
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)  # 解码器每一步的输入 seq-1个[1, batch, embed_size]
            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]

            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    state = first_state  # 解码器初始状态
                decoder_input = decoder_inputs[idx]  # 当前时间步输入 [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                assert output.squeeze().equal(state[0][-1])
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab, _mu, _logvar, mu, logvar
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            batch_size = id_posts.size(0)

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]
            # state = [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            if isinstance(state_posts, tuple):  # 如果是lstm则取h
                state_posts = state_posts[0]  # [layers, batch, dim]
            x = state_posts[-1, :, :]  # 取最后一层 [batch, dim]

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            # 重参数化
            z = _mu + (0.5 * _logvar).exp() * sampled_latents  # [batch, latent]

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
            done = torch.tensor([0] * batch_size).bool()
            first_input_id = (torch.ones((1, batch_size)) * self.model_config.start_id).long()
            if gpu:
                done = done.cuda()
                first_input_id = first_input_id.cuda()

            outputs = []
            for idx in range(max_len):
                if idx == 0:  # 第一个时间步
                    state = first_state  # 解码器初始状态
                    decoder_input = self.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                else:
                    decoder_input = self.embedding(next_input_id)  # [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.model_config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq, num_vocab]

            return output_vocab, _mu, _logvar, None, None

    def print_parameters(self):
        r""" 统计参数 """
        total_num = 0  # 参数总数
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"参数总数: {total_num}")

    def save_model(self, running_avg_ppl, epoch, global_step):
        r""" 保存模型 """
        state = {'embedding': self.embedding.state_dict(),
                 'post_encoder': self.post_encoder.state_dict(),
                 'response_encoder': self.response_encoder.state_dict(),
                 'prior_net': self.prior_net.state_dict(),
                 'recognize_net': self.recognize_net.state_dict(),
                 'prepare_state': self.prepare_state.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'projector': self.projector.state_dict(),
                 'epoch': epoch,
                 'global_step': global_step}

        model_save_path = os.path.join(self.model_dir,
                                       'model_{:.3f}_{}_{}'.format(running_avg_ppl, epoch, global_step))
        torch.save(state, model_save_path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path, map_location=torch.device('cuda' if config.USE_CUDA else 'cpu'))
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.response_encoder.load_state_dict(checkpoint['response_encoder'])
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.prepare_state.load_state_dict(checkpoint['prepare_state'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print('载入模型完成')
        return epoch

    def train_one_batch(self, batch, train=True):
        output_vocab, _mu, _logvar, mu, logvar = self.forward(batch, gpu=config.USE_CUDA)  # 前向传播
        outputs = (output_vocab, _mu, _logvar, mu, logvar)
        labels = batch['responses'][:, 1:]  # 去掉start_id
        masks = batch['masks']
        loss, nll_loss, kld_loss, ppl, kld_weight = self.compute_loss(outputs, labels, masks, self.global_step)  # 计算损失
        loss = loss.mean()
        ppl = ppl.mean().exp()
        nll_loss = nll_loss.mean()
        kld_loss = kld_loss.mean()
        if train:
            loss.backward()  # 反向传播
            self.optim.step()  # 更新参数
            self.optim.optimizer.zero_grad()  # 清空梯度

        return loss.item(), ppl.item(), loss, nll_loss.item(), kld_loss.item(), kld_weight

    def compute_loss(self, outputs, labels, masks, global_step):
        def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):  # [batch, latent]
            """ 两个高斯分布之间的kl散度公式 """
            kld = 0.5 * torch.sum(prior_logvar - recog_logvar - 1
                                  + recog_logvar.exp() / prior_logvar.exp()
                                  + (prior_mu - recog_mu).pow(2) / prior_logvar.exp(), 1)
            return kld  # [batch]

        # output_vocab: [batch, len_decoder, num_vocab] 对每个单词的softmax概率
        output_vocab, _mu, _logvar, mu, logvar = outputs  # 先验的均值、log方差，后验的均值、log方差

        token_per_batch = masks.sum(1)  # 每个样本要计算损失的token数 [batch]
        len_decoder = masks.size(1)  # 解码长度

        output_vocab = output_vocab.reshape(-1, self.vocab.n_words)  # [batch*len_decoder, num_vocab]
        labels = labels.reshape(-1)  # [batch*len_decoder]
        masks = masks.reshape(-1)  # [batch*len_decoder]

        # nll_loss需要自己求log，它只是把label指定下标的损失取负并拿出来，reduction='none'代表只是拿出来，而不需要求和或者求均值
        _nll_loss = F.nll_loss(output_vocab.clamp_min(1e-12).log(), labels,
                               reduction='none')  # 每个token的-log似然 [batch*len_decoder]
        _nll_loss = _nll_loss * masks  # 忽略掉不需要计算损失的token [batch*len_decoder]

        nll_loss = _nll_loss.reshape(-1, len_decoder).sum(1)  # 每个batch的nll损失 [batch]

        ppl = nll_loss / token_per_batch.clamp_min(1e-12)  # ppl的计算需要平均到每个有效的token上 [batch]

        # kl散度损失 [batch]
        kld_loss = gaussian_kld(mu, logvar, _mu, _logvar)

        # kl退火
        kld_weight = min(1.0 * global_step / self.model_config.kl_step, 1)  # 一次性退火
        # kld_weight = min(1.0 * (global_step % (2 * self.model_config.kl_step)) / self.model_config.kl_step, 1)  # 周期性退火

        # 损失
        loss = nll_loss + kld_weight * kld_loss

        return loss, nll_loss, kld_loss, ppl, kld_weight

    def evaluate(self, data, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", verbose=False):
        dial, ref, hyp_b = [], [], []
        # t = Translator(model, model.vocab)

        l = []
        p = []
        ent_b = []

        pbar = tqdm(enumerate(data), total=len(data))
        for j, batch in pbar:
            loss, ppl, _ = self.train_one_batch(batch, train=False)
            l.append(loss)
            p.append(ppl)
            if ((j < 3 and ty != "test") or ty == "test"):

                # sent_b, _ = t.translate_batch(batch)
                output_vocab, _, _, _, _ = self.forward(batch, inference=True, max_len=60, gpu=config.USE_CUDA)
                sent_b = output_vocab.argmax(2).detach().tolist()

                for i in range(len(batch["target_txt"])):
                    new_words = []
                    for w in sent_b[i]:
                        if w == config.EOS_idx:
                            break
                        new_words.append(w)
                        if len(new_words) > 2 and (new_words[-2] == w):
                            new_words.pop()

                    sent_beam_search = ' '.join([self.vocab.index2word[idx] for idx in new_words])
                    hyp_b.append(sent_beam_search)
                    ref.append(batch["target_txt"][i])
                    dial.append(batch['input_txt'][i])
                    ent_b.append(
                        self.bert.predict_label([sent_beam_search for _ in range(len(batch['persona_txt'][i]))],
                                                batch['persona_txt'][i]))

            pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l), np.mean(p)))
            if (j > 4 and ty == "train"): break
        loss = np.mean(l)
        ppl = np.mean(p)
        ent_b = np.mean(ent_b)
        bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)

        if (verbose):
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")
            print_all(dial, ref, hyp_b, max_print=3 if ty != "test" else 100000000)
            print("EVAL\tLoss\tPeplexity\tEntl_b\tBleu_b")
            print("{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(ty, loss, ppl, ent_b, bleu_score_b))
        return loss, ppl, ent_b, bleu_score_b


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
