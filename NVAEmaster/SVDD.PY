import argparse
import importlib.util
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import get_loaders_com




def select_normal_mask(batch, normal_key: str, normal_value: float):
    if not normal_key or normal_key not in batch:
        return None
    y = batch[normal_key]
    if torch.is_tensor(y):
        return (y == normal_value)
    return torch.tensor([v == normal_value for v in y], dtype=torch.bool)


def encoder_mu(nvae, x):
    # deterministic encoder mapping mu(x;W) for SVDD objective
    s = nvae.stem(2 * x - 1.0)
    for cell in nvae.pre_process:
        s = cell(s)
    for cell in nvae.enc_tower:
        if cell.cell_type != "combiner_enc":
            s = cell(s)
    ftr = nvae.enc0(s)
    z = F.adaptive_avg_pool2d(ftr, (1, 1)).flatten(1)
    return z


def enable_dropout_only(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.train()


def _reduce_batch_mean(t: torch.Tensor):
    if t.dim() <= 1:
        return t.mean()
    dims = tuple(range(1, t.dim()))
    return t.mean(dim=dims).mean()


def _forward_with_latent(nvae, x):
    """
    Returns:
        z: [B, D] latent vector
        logits: decoder logits
        log_q: [B] (approx posterior log prob)
        log_p: [B] (prior log prob)
    """
    try:
        out = nvae(x, return_latents=True)
        if isinstance(out, (list, tuple)) and len(out) == 6:
            logits, log_q, log_p, _, _, z_groups = out
            z_parts = [F.adaptive_avg_pool2d(zg, (1, 1)).flatten(1) for zg in z_groups]
            z = torch.cat(z_parts, dim=1)
            return z, logits, log_q, log_p
    except TypeError:
        pass

    # Fallback if model.py was not updated with return_latents=True.
    logits, log_q, log_p, _, _ = nvae(x)
    z = encoder_mu(nvae, x)
    return z, logits, log_q, log_p


@torch.no_grad()
def init_center_c(nvae, loader, device, normal_key, normal_value):
    n = 0
    c = None
    nvae.eval()
    for batch in tqdm(loader, desc="init_center"):
        x = batch["image"].to(device)
        m = select_normal_mask(batch, normal_key, normal_value)
        if m is not None:
            m = m.to(x.device)
            if m.sum().item() == 0:
                continue
            x = x[m]

        z, _, _, _ = _forward_with_latent(nvae, x)
        if c is None:
            c = torch.zeros(z.size(1), device=device)
        c += z.sum(dim=0)
        n += z.size(0)

    if n == 0:
        raise RuntimeError("No samples found to initialize center c.")
    c /= n
    return c


def train_svdd(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    nvae = build_nvae(args.nvae_ckpt, device)

   

    train_loader, _, _, _ = get_loaders_com(
        dataset="medical",
        args=argparse.Namespace(data=args.data, batch_size=args.batch_size),
        tokenizer=tokenizer,
    )

    # train NVAE
    for p in nvae.parameters():
        p.requires_grad = True

    params = [p for p in nvae.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    c = init_center_c(nvae, train_loader, device, args.normal_key, args.normal_value)

    for ep in range(args.epochs):
        # Keep BN statistics stable; activate dropout explicitly for MC passes.
        nvae.eval()
        nvae.apply(enable_dropout_only)
        total = 0.0
        steps = 0

        for batch in tqdm(train_loader, desc=f"svdd_ep{ep+1}"):
            x = batch["image"].to(device)
            m = select_normal_mask(batch, args.normal_key, args.normal_value)
            if m is not None:
                m = m.to(x.device)
                if m.sum().item() == 0:
                    continue
                x = x[m]

            rec_kl = 0.0
            svdd_term = 0.0
            z_samples = []
            xhat_samples = []

            for _ in range(args.mc_passes):
                z, logits, log_q, log_p = _forward_with_latent(nvae, x)

                # Reconstruction term: ||x - x_hat||^2 (MSE-style)
                dec = nvae.decoder_output(logits)
                if hasattr(dec, "mean"):
                    xhat = dec.mean()
                else:
                    xhat = torch.clamp(dec.dist.mu, -1, 1.0) / 2.0 + 0.5
                rec = F.mse_loss(xhat, x, reduction="mean")

                # KL term
                d_kl = (log_q - log_p).mean()

                # SVDD compactness
                svdd = ((z - c) ** 2).sum(dim=1).mean()

                rec_kl = rec_kl + (rec + args.beta_kl * d_kl)
                svdd_term = svdd_term + svdd
                z_samples.append(z)
                xhat_samples.append(xhat)

            rec_kl = rec_kl / float(args.mc_passes)
            svdd_term = svdd_term / float(args.mc_passes)

            # Uncertainty regularization: Var[z] + Var[x_hat]
            z_stack = torch.stack(z_samples, dim=0)     # [T, B, D]
            x_stack = torch.stack(xhat_samples, dim=0)  # [T, B, C, H, W]
            var_z = z_stack.var(dim=0, unbiased=False).mean()
            var_x = x_stack.var(dim=0, unbiased=False).mean()

            loss = rec_kl + args.lambda_svdd * svdd_term + args.delta_unc * (var_z + var_x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            steps += 1

        print(f"[SVDD-MC] epoch {ep+1}: loss={total / max(1, steps):.6f}")

    out = {
        "nvae_state_dict": nvae.state_dict(),
        "center_c": c.detach().cpu(),
        "normal_key": args.normal_key,
        "normal_value": args.normal_value,
    }
    torch.save(out, args.save)
    print(f"saved -> {args.save}")



def build_parser():
    p = argparse.ArgumentParser()
    t = p.add_argument_group("train")
    p.add_argument("--nvae_ckpt", default="/public/ATIQA/experiments/eval-exp/checkpoint.pt")
    p.add_argument("--data", default="/public/ATIQA/Datasets/iu_xray/")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--mc_passes", type=int, default=5)
    p.add_argument("--beta_kl", type=float, default=1.0)
    p.add_argument("--lambda_svdd", type=float, default=1.0)
    p.add_argument("--delta_unc", type=float, default=0.1)
    p.add_argument("--normal_key", default="")
    p.add_argument("--normal_value", type=float, default=0.0)
    p.add_argument("--save", default="/public/ATIQA/multimodel_bridge/svdd_oneclass.pt")
    return p


def main():
    args = build_parser().parse_args()
    train_svdd(args)


if __name__ == "__main__":
    main()

def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging, args, c):
    nelbo = utils.AvgrageMeter()
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

            rec_kl = 0.0
            svdd_term = 0.0
            z_samples, xhat_samples = [], []
            recon_iter = 0.0
            kl_iter = 0.0
            xhat_last = None

            for _ in range(args.mc_passes):
                # requires model.forward(..., return_latents=True)
                logits, log_q, log_p, _, _, z_groups = model(x, return_latents=True)
                z = torch.cat(
                    [F.adaptive_avg_pool2d(zg, (1, 1)).flatten(1) for zg in z_groups],
                    dim=1
                )

                output = model.decoder_output(logits)
                xhat = output.mean() if hasattr(output, "mean") else output.sample()

                rec = F.mse_loss(xhat, x, reduction="mean")
                d_kl = (log_q - log_p).mean()
                svdd = ((z - c) ** 2).sum(dim=1).mean()

                rec_kl += (rec + args.beta_kl * d_kl)
                svdd_term += svdd
                z_samples.append(z)
                xhat_samples.append(xhat)

                recon_iter += rec
                kl_iter += d_kl
                xhat_last = xhat

            rec_kl = rec_kl / float(args.mc_passes)
            svdd_term = svdd_term / float(args.mc_passes)
            recon_iter = recon_iter / float(args.mc_passes)
            kl_iter = kl_iter / float(args.mc_passes)

            var_z = torch.stack(z_samples, 0).var(dim=0, unbiased=False).mean()
            var_x = torch.stack(xhat_samples, 0).var(dim=0, unbiased=False).mean()

            loss = rec_kl + args.lambda_svdd * svdd_term + args.delta_unc * (var_z + var_x)

            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()

            if args.weight_decay_norm_anneal:
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
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
                output_img = xhat_last[:n*n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)
            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()['param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', kl_iter, global_step)
            writer.add_scalar('train/recon_iter', recon_iter, global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step
