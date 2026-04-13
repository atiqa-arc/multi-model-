import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from model_bridge import LatentMemoryBridgeModel
BIOGPT_ROOT = Path("/public/ATIQA/BioGPTLLM")
sys.path.insert(0, str(BIOGPT_ROOT))
from src.dataset import get_loaders_com


def configure_trainable_params(model, stage, train_last_n_layers=1, train_lm_head=False):
    for p_ in model.nvae.parameters():
        p_.requires_grad = False

    for p_ in model.adapter.parameters():
        p_.requires_grad = True

    for p_ in model.decoder.parameters():
        p_.requires_grad = False

    for p_ in model.decoder.cross_attn.parameters():
        p_.requires_grad = True

    model.decoder.cross_attn_gate.requires_grad = True

    for p_ in model.decoder.cross_attn_norm.parameters():
        p_.requires_grad = True

    if stage == "stage2":
        last_n = max(1, int(train_last_n_layers))
        for layer in model.decoder.model.layers[-last_n:]:
            for p_ in layer.parameters():
                p_.requires_grad = True

        if train_lm_head and hasattr(model.decoder, "lm_head"):
            for p_ in model.decoder.lm_head.parameters():
                p_.requires_grad = True


def print_trainable_params(model):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"trainable parameter tensors: {len(names)}")
    for name in names:
        print(f"  {name}")

def train(model, loader, optimizer, device, grad_accum_steps):
    model.train()
    model.nvae.eval()

    total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train"), start=1):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model(images, input_ids, attn)
            loss = out.loss / grad_accum_steps

        if torch.isnan(out.loss) or torch.isinf(out.loss):
            print(f"Bad loss detected at step {step}")
            if hasattr(out, "logits"):
                print("logits nan:", torch.isnan(out.logits).any().item())
                print("logits inf:", torch.isinf(out.logits).any().item())
                print("logits abs max:", out.logits.detach().abs().max().item())
            break

        loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += out.loss.item()

    return total / max(1, len(loader))



@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    model.nvae.eval()

    total = 0.0
    for batch in tqdm(loader, desc="valid"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model(images, input_ids, attn)

        total += out.loss.item()

    return total / max(1, len(loader))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nvae_ckpt", default="/public/ATIQA/NVAEmaster/eval-exp/checkpoint.pt")
    p.add_argument("--data", default="/public/ATIQA/Datasets/iu_xray/")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--bridge_type", choices=["baseline", "latent_memory"], default="latent_memory")
    p.add_argument("--mc_passes", type=int, default=10)
    p.add_argument("--text_model_name", default="/public/model/Llama-3.2-3B-Instruct")
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    p.add_argument("--saved", default="")
    p.add_argument("--train_last_n_layers", type=int, default=1)
    p.add_argument("--train_lm_head", action="store_true")
    args = p.parse_args()
    if not args.saved:
        if args.stage == "stage2":
            args.saved = "/public/ATIQA/multimodel_bridge/llama_stage2.pt"
        else:
            args.saved = "/public/ATIQA/multimodel_bridge/llama_stage1.pt"
    saved = Path(args.saved)
    saved.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = LatentMemoryBridgeModel(
        nvae_ckpt=args.nvae_ckpt,
        text_model_name=args.text_model_name,
        mc_passes=args.mc_passes
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded init checkpoint from {args.init_ckpt}")

    configure_trainable_params(
        model,
        stage=args.stage,
        train_last_n_layers=args.train_last_n_layers,
        train_lm_head=args.train_lm_head,
    )
    print_trainable_params(model)

    train_loader, valid_loader, _, _ = get_loaders_com(
        dataset="medical",
        args=argparse.Namespace(data=args.data, batch_size=args.batch_size),
        tokenizer=model.tokenizer,
    )

    params = [x for x in model.parameters() if x.requires_grad]
    opt = optim.AdamW(params, lr=args.lr)

    best_val = float("inf")
    epochs_without_improvement = 0
    min_delta = 1e-4

    for ep in range(args.epochs):
        tr = train(model, train_loader, opt, device, args.grad_accum_steps)
        va = validate(model, valid_loader, device)
        print(f"epoch {ep+1}: train_loss={tr:.4f} val_loss={va:.4f}")

        if va < (best_val - min_delta):
            best_val = va
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "bridge_type": args.bridge_type,
                "mc_passes": args.mc_passes,
                "text_model_name": args.text_model_name,
                "stage": args.stage,
            }, args.saved)
            print(f"saved best val -> {args.saved}")
        else:
            epochs_without_improvement += 1
            print(f"no val improvement: {epochs_without_improvement}/{args.patience}")
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {ep+1}")
                break


if __name__ == "__main__":
    main()
