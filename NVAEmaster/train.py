# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import datasets
import utils
from fid.fid_score import calculate_frechet_distance, compute_statistics_of_generator, load_statistics
from fid.inception import InceptionV3
from model import AutoEncoder
from thirdparty.adamax import Adamax
from torchvision.utils import save_image


def extract_mu_latent(model, x):
    logits, log_q, log_p, kl_all, kl_diag, _, mu_groups = model(
        x, return_latents=True, return_posterior_means=True
    )
    mu = torch.cat(
        [F.adaptive_avg_pool2d(mu_g, (1, 1)).flatten(1) for mu_g in mu_groups],
        dim=1
    )
    return logits, log_q, log_p, kl_all, kl_diag, mu


@torch.no_grad()
def init_svdd_center(train_queue, model, args, logging):
    model.eval()
    center_sum = None
    num_samples = 0

    for batch in train_queue:
        x = batch[0] if len(batch) > 1 else batch
        x = x.cuda(non_blocking=True)
        x = utils.pre_process(x, args.num_x_bits)

        _, _, _, _, _, mu = extract_mu_latent(model, x)
        if center_sum is None:
            center_sum = torch.zeros(mu.size(1), device=mu.device, dtype=mu.dtype)
        center_sum += mu.sum(dim=0)
        num_samples += mu.size(0)

    if center_sum is None or num_samples == 0:
        raise RuntimeError('Unable to initialize SVDD center: no training samples were found.')

    if args.distributed:
        dist.all_reduce(center_sum, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor(float(num_samples), device=center_sum.device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        num_samples = int(count_tensor.item())

    center = center_sum / float(num_samples)
    model.train()
    logging.info('Initialized fixed SVDD center from %d training samples.', num_samples)
    return center.detach()



def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)
    recon_dir = os.path.join(args.save, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)
    args.recon_dir = recon_dir


    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    nvae_model = AutoEncoder(args, writer, arch_instance).cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(nvae_model))
    logging.info('groups per scale: %s, total_groups: %d', nvae_model.groups_per_scale, sum(nvae_model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(nvae_model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(nvae_model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    center_c = None
    score_threshold = None
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        nvae_model.load_state_dict(checkpoint['state_dict'])
        nvae_model = nvae_model.cuda()


        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        if 'svdd_center' in checkpoint:
            center_c = checkpoint['svdd_center'].cuda()
        if 'score_threshold' in checkpoint:
            score_threshold = float(checkpoint['score_threshold'])
    else:
        global_step, init_epoch = 0, 0

    if center_c is None:
        center_c = init_svdd_center(train_queue, nvae_model, args, logging)
 
    best_train_nelbo = float('inf')
    best_val_nelbo = float('inf')
    epochs_without_improvement = 0
    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step, score_threshold = train(
            train_queue, nvae_model, cnn_optimizer, grad_scalar, global_step,
            warmup_iters, writer, logging, args, center_c
        )
        logging.info('train_nelbo %f', train_nelbo)
        logging.info('train score threshold (%0.1f%%) %f', args.score_percentile, score_threshold)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)
        writer.add_scalar('train/score_threshold', score_threshold, global_step)
        if train_nelbo < (best_train_nelbo - args.min_delta):
            best_train_nelbo = train_nelbo
            if args.global_rank == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': nvae_model.state_dict(),
                    'optimizer': cnn_optimizer.state_dict(),
                    'global_step': global_step,
                    'args': args,
                    'arch_instance': arch_instance,
                    'scheduler': cnn_scheduler.state_dict(),
                    'grad_scalar': grad_scalar.state_dict(),
                    'svdd_center': center_c.detach().cpu(),
                    'score_threshold': score_threshold,
                    'best_train_nelbo': best_train_nelbo,
                    'best_val_nelbo': best_val_nelbo,
                }, checkpoint_file)
                logging.info('Saved best by train_nelbo: %f', train_nelbo)


        nvae_model.eval()
        did_validation = False
        # Validate every epoch.
        eval_freq = 1
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            with torch.no_grad():
                num_samples = 16
                n = int(np.floor(np.sqrt(num_samples)))
                for t in [0.7, 0.8, 0.9, 1.0]:
                    logits = nvae_model.sample(num_samples, t)
                    output = nvae_model.decoder_output(logits)
                    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample(t)
                    output_tiled = utils.tile_image(output_img, n)
                    writer.add_image('generated_%0.1f' % t, output_tiled, global_step)

            valid_neg_log_p, valid_nelbo = test(valid_queue, nvae_model, num_samples=10, args=args, logging=logging)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)
            did_validation = True

        if did_validation:
            if valid_nelbo < (best_val_nelbo - args.min_delta):
                best_val_nelbo = valid_nelbo
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info('No val improvement: %d/%d', epochs_without_improvement, args.patience)
                if epochs_without_improvement >= args.patience:
                    logging.info('Early stopping at epoch %d', epoch)
                    break

    # Final validation
    valid_neg_log_p, valid_nelbo = test(valid_queue, nvae_model, num_samples=1000, args=args, logging=logging)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.close()
#training logic & losses
def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging, args, center_c):
    alpha_i = utils.kl_balancer_coeff(
        num_scales=model.num_latent_scales,
        groups_per_scale=model.groups_per_scale,
        fun='square'
    )

    nelbo = utils.AvgrageMeter()
    score_values = []
    model.train()

    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        x = utils.pre_process(x, args.num_x_bits)

        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()

        with autocast():
            kl_coeff = utils.kl_coeff(
                global_step,
                args.kl_anneal_portion * args.num_total_iter,
                args.kl_const_portion * args.num_total_iter,
                args.kl_const_coeff
            )

            loss_mc = 0.0
            svdd_mc = 0.0
            mu_samples = []
            xhat_samples = []

            recon_iter = 0.0
            kl_iter = 0.0
            svdd_iter = 0.0
            output_last = None
            kl_diag_last = None
            kl_coeffs_last = None
            kl_vals_last = None

            for _ in range(args.mc_passes):
                logits, log_q, log_p, kl_all, kl_diag, mu = extract_mu_latent(model, x)

                output = model.decoder_output(logits)

                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(
                    kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i
                )

                nelbo_batch = recon_loss + balanced_kl
                loss_one = torch.mean(nelbo_batch)
                loss_mc = loss_mc + loss_one

                svdd_loss = torch.sum((mu - center_c.unsqueeze(0)) ** 2, dim=1).mean()
                svdd_mc = svdd_mc + svdd_loss
                mu_samples.append(mu)

                if hasattr(output, "mean") and callable(output.mean):
                    xhat = output.mean()
                elif hasattr(output, "mean"):
                    xhat = output.mean
                else:
                    xhat = output.sample()

                xhat_samples.append(xhat)

                recon_iter = recon_iter + torch.mean(recon_loss)
                kl_iter = kl_iter + torch.mean(sum(kl_all))
                svdd_iter = svdd_iter + svdd_loss

                output_last = output
                kl_diag_last = kl_diag
                kl_coeffs_last = kl_coeffs
                kl_vals_last = kl_vals

            loss_mc = loss_mc / float(args.mc_passes)
            recon_iter = recon_iter / float(args.mc_passes)
            kl_iter = kl_iter / float(args.mc_passes)
            svdd_mc = svdd_mc / float(args.mc_passes)
            svdd_iter = svdd_iter / float(args.mc_passes)

            mu_stack = torch.stack(mu_samples, dim=0)      # [T, B, D]
            xhat_stack = torch.stack(xhat_samples, dim=0)  # [T, B, C, H, W]
            mu_mean = mu_stack.mean(dim=0)
            xhat_mean = xhat_stack.mean(dim=0)

            var_z = mu_stack.var(dim=0, unbiased=False).mean()
            var_xhat = xhat_stack.var(dim=0, unbiased=False).mean()

            uncertainty_loss = var_z + var_xhat

            re_batch = torch.mean((x - xhat_mean) ** 2, dim=(1, 2, 3))
            svdd_batch = torch.sum((mu_mean - center_c.unsqueeze(0)) ** 2, dim=1)
            unc_batch = mu_stack.var(dim=0, unbiased=False).mean(dim=1) + \
                xhat_stack.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
            fused_score_batch = (
                args.alpha_score * re_batch +
                args.beta_score * svdd_batch +
                args.gamma_score * unc_batch
            )
            score_values.extend(fused_score_batch.detach().cpu().tolist())
            fused_score_iter = fused_score_batch.mean()

            loss = loss_mc + args.lambda_svdd * svdd_mc + args.delta_unc * uncertainty_loss

            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()

            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, \
                    'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + \
                            kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss = loss + norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 1000 == 0:
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n*n]
                output_img = output_last.mean if isinstance(output_last, torch.distributions.bernoulli.Bernoulli) else output_last.sample()


                
                output_img = output_img[:n*n]
               
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
               

                
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            
                save_path = os.path.join(args.recon_dir, f'recon_step_{global_step+1}.png')
                save_image(in_out_tiled, save_path)
                

            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)
            writer.add_scalar('train/uncertainty_var_z', var_z, global_step)
            writer.add_scalar('train/uncertainty_var_xhat', var_xhat, global_step)
            writer.add_scalar('train/uncertainty_total', uncertainty_loss, global_step)
            writer.add_scalar('train/svdd_iter', svdd_iter, global_step)
            writer.add_scalar('train/fused_score_iter', fused_score_iter, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()['param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', kl_iter, global_step)
            writer.add_scalar('train/recon_iter', recon_iter, global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)

            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag_last):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs_last[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals_last[i], global_step)

            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1

    score_threshold = float(np.percentile(score_values, args.score_percentile)) if score_values else 0.0
    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step, score_threshold

#evulation function  
def test(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(valid_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, _ = model(x)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg

#generate synthetic images from the trained VAE.images are generated batch-by-batch.fully synthetic images
def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()

# generated image from model checkvalidation by using  traning dataset
def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid




# traning loop is started
if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    parser.add_argument('--mc_passes', type=int, default=5)

    # experimental results

    parser.add_argument('--root', type=str, default='/public/ATIQA/NVAEmaster',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='medical',
                        choices=['medical'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/public/ATIQA/Datasets/iu_xray/',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=0,
                    help='number of dataloader worker processes')

    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                    help='early stopping patience (in validation checks)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                    help='minimum val improvement to reset patience')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    parser.add_argument('--beta_kl', type=float, default=1.0)
    parser.add_argument('--lambda_svdd', type=float, default=0.1)
    parser.add_argument('--delta_unc', type=float, default=0.01)
    parser.add_argument('--alpha_score', type=float, default=1.0,
                        help='weight for reconstruction error in fused anomaly score')
    parser.add_argument('--beta_score', type=float, default=1.0,
                        help='weight for SVDD distance in fused anomaly score')
    parser.add_argument('--gamma_score', type=float, default=1.0,
                        help='weight for uncertainty in fused anomaly score')
    parser.add_argument('--score_percentile', type=float, default=95.0,
                        help='percentile of training fused scores used as anomaly threshold')
                   
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=5,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=10,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    parser.add_argument('--mc_dropout_p', type=float, default=0.1)

    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--mc_dropout_dec_p', type=float, default=0.1)
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.

    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    
    args.distributed = False
    args.local_rank = 0
    args.global_rank = 0

    print('starting in single-GPU mode')
    main(args)
    sys.exit(0)


    
