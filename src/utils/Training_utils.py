import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import sklearn.preprocessing
import sklearn.metrics

from tqdm import tqdm
from tensorboardX import SummaryWriter

eps = np.finfo(float).eps


def generator(dataloader):
    while True:
        for data in dataloader:
            yield data


def mask_column(X, col_idx):
    assert len(X.shape) == 2
    ret = X.clone()
    ret[:, col_idx] = 0
    return ret


def add_masked_noise(x, std, col_idx):
    normal = torch.distributions.Normal(0, std)
    batch, dim = x.shape
    ret_x = x.clone()
    for idx in col_idx:
        ret_x[:, idx] += normal.sample([batch, ])
    return ret_x


def calc_accracy_softmax(y, out):
    label = y.flatten().cpu().detach().numpy().astype(np.int)
    pred = torch.argmax(out, dim=-1).cpu().detach().numpy().astype(np.int)
    return sklearn.metrics.accuracy_score(label, pred)


def calc_accracy_sigmoid(y, out):
    label = y.flatten().cpu().detach().numpy().astype(np.int)
    pred = (torch.sigmoid(out).cpu().detach().numpy() > 0.5).astype(np.int)
    return sklearn.metrics.accuracy_score(label, pred)


def add_noise_on_pixel(x, pixel_mask, distribution):
    noise = distribution.sample(x.size())
    batch = x.shape[0]
    ret = x.clone()
    ret += pixel_mask.repeat(batch, 1).type(torch.FloatTensor)*noise
    return ret


# handle the dimension error, the multi-class require input : (N, C), label : (N)
def warp_loss_fn(loss_criterion, x, y):
    if type(loss_criterion) is nn.CrossEntropyLoss:
        return loss_criterion(x, y.flatten())
    else:
        return loss_criterion(x, y)


def train(
        model, opt, src_loss_criterion, train_dataloader, val_dataloader,
        alpha, beta, gamma, epochs, noise_fn, metric_fn, log_name, feature_names, K):
    train_G = generator(train_dataloader)
    val_G = generator(val_dataloader)
    writer = SummaryWriter('./logs/%s-a%f,b%f,g%f' % (log_name, alpha, beta, gamma))
    iters = 1
    with tqdm(total=epochs*len(train_dataloader)) as pbar:
        for epoch in range(epochs):
            for _ in range(len(train_dataloader)):
                x, y = next(train_G)
                noised_x = noise_fn(x)
                val_x, val_y = next(val_G)
                val_noised_x = noise_fn(val_x)

                model.train()
                train_out = model(x)
                reg_loss, w_loss, entropy_loss = model.calc_reg_loss(F.mse_loss)

                src_loss = warp_loss_fn(src_loss_criterion, train_out, y)
                loss = alpha*reg_loss + beta*w_loss + gamma*entropy_loss + src_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    model.eval()

                    val_out = model(val_x)
                    val_src_loss = warp_loss_fn(src_loss_criterion, val_out, val_y)

    #                 noised
                    noised_train_out = model(noised_x)
                    noised_val_out = model(val_noised_x)
                    noised_src = warp_loss_fn(src_loss_criterion, noised_train_out, y)
                    val_noised_src = warp_loss_fn(src_loss_criterion, noised_val_out, val_y)
    #                 acc
                    train_acc = metric_fn(y, train_out)
                    val_acc = metric_fn(val_y, val_out)
                    noised_train_acc = metric_fn(y, noised_train_out)
                    noised_val_acc = metric_fn(val_y, noised_val_out)

                pbar.update(1)
                w_arr = model.w.cpu().detach().numpy().flatten()
                w_prine = torch.sigmoid(model.w).cpu().detach().numpy().flatten()
                w_ratio = model.select_lay.calc_ratio().cpu().detach().numpy().flatten()
                sorted_ratio = sorted([(feature_names[i], a) for i, a in enumerate(w_ratio)], key=lambda a: a[1])
#                 display the top K feature and lowest K
                if len(feature_names) <= 2*K:
                    buf = sorted_ratio
                else:
                    buf = sorted_ratio[:K] + sorted_ratio[-K:]
                buf = ['%2s:%6.2f' % (name, a*1000) for name, a in buf]
                buf = ','.join(['%50s' % s for s in buf])
                buf_str = 'acc : %10.3f, val_acc : %10.3f, loss: %10.3f, \
                             val_loss: %10.4f, w_loss : %10.3f, entropy : %10.3f, \
                             regularizer : %10.3f %400s' % (
                                         train_acc.item(), val_acc.item(),
                                         src_loss.item(), val_src_loss.item(),
                                         w_loss.item(), entropy_loss.item(),
                                         reg_loss.item(), buf.encode('ascii', 'ignore'))
                pbar.set_postfix_str(buf_str)
                if epoch >= 0 and iters % 10 == 0:
                    writer.add_scalars(
                            'data/loss',
                            {'train': loss.item()},
                            iters
                            )
                    writer.add_scalars(
                            'data/src_loss',
                            {
                                'train': src_loss.item(),
                                'validation': val_src_loss.item()
                            },
                            iters
                            )
                    writer.add_scalars(
                            'data/noised_loss',
                            {
                                'train': noised_src.item(),
                                'validation': val_noised_src.item()
                            },
                            iters)
                    writer.add_scalars(
                            'data/accuracy',
                            {
                                'train': train_acc.item(),
                                'validation': val_acc.item(),
                                'noised_train': noised_train_acc.item(),
                                'noised_validation': noised_val_acc.item(),
                            },
                            iters)
                    writer.add_scalars(
                            'data/w_loss',
                            {
                                'train': w_loss.item()
                            },
                            iters)
                    writer.add_scalars(
                            'data/entropy',
                            {
                                'train': entropy_loss.item()
                            },
                            iters)
                    writer.add_scalars(
                            'data/reg_loss',
                            {
                                'train': reg_loss.item()
                            },
                            iters)
                    writer.add_scalars(
                            'data/w_value',
                            {
                                '%s' % feature_names[i]: v for i, v in enumerate(w_arr)
                            },
                            iters)
                    writer.add_scalars(
                            'data/w_prine',
                            {
                                '%s' % feature_names[i]: v for i, v in enumerate(w_prine)
                            },
                            iters)
                    writer.add_scalars(
                            'data/w_ratio',
                            {
                                '%s' % feature_names[i]: v for i, v in enumerate(w_ratio)
                            },
                            iters)
                    writer.add_scalars(
                            'data/w_z',
                            {
                                'nothing': 0
                            },
                            iters)

                iters += 1
    writer.close()

    print 'done'


print "^((?!.*((w_prine)|(w_value)).*).)*$"
