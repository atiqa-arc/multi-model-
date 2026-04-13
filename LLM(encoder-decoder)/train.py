import argparse
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from src.dataset import get_loaders_com
from src.model import EncoderDecoder

def validate(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for batch in valid_loader:
            images = batch["image"]
            captions = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(images, captions)
            total_loss += outputs.loss.item()
    return total_loss / len(valid_loader)

def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    epochs,
    gradient_accumulation_steps,
    max_grad_norm,
    patience,
    device,
    train_model_file,
):
    best_train_loss = float("inf")
    best_train_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    min_delta = 1e-4

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        ):
            images = batch["image"]
            captions = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            outputs = model(images, captions)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # handle leftover grads if steps not divisible by accumulation
        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, valid_loader, device)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # save best checkpoint by training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch + 1
            torch.save(model.state_dict(), train_model_file)
            print(f"Model saved to {train_model_file}")

        # early stopping by validation loss
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No val improvement for {epochs_without_improvement}/{patience} epochs")
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best training loss: {best_train_loss:.6f} at epoch {best_train_epoch}")
    print(f"Best validation loss used for early stopping: {best_val_loss:.6f}")

class TestEvaluator:
    def __init__(self, model, tokenizer, test_loader, device, eval_max_length=32, num_beams=2):
        self.model = model
        self.tokenizer = tokenizer
        self.test_loader = test_loader
        self.device = device
        self.eval_max_length = eval_max_length
        self.num_beams = num_beams

    def compute_corpus_bleu(self, references, hypotheses):
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        refs = [[word_tokenize(r.lower())] for r in references]
        hyps = [word_tokenize(h.lower()) for h in hypotheses]
        smoothing = SmoothingFunction().method1

        bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = corpus_bleu(refs, hyps, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing)
        bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        return bleu1, bleu2, bleu3, bleu4

    def compute_sentence_rouge_l(self, references, hypotheses):
        from rouge_score import rouge_scorer

        scores = []
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        for ref, hyp in zip(references, hypotheses):
            score = scorer.score(ref, hyp)["rougeL"].fmeasure
            scores.append(score)
        return np.mean(scores)

    def compute_sentence_meteor(self, references, hypotheses):
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score

        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = word_tokenize(ref.lower())
            hyp_tokens = word_tokenize(hyp.lower())
            scores.append(meteor_score([ref_tokens], hyp_tokens))
        return np.mean(scores)

    def evaluate_reports(self, references, hypotheses):
        bleu1, bleu2, bleu3, bleu4 = self.compute_corpus_bleu(references, hypotheses)
        rouge_l = self.compute_sentence_rouge_l(references, hypotheses)
        meteor = self.compute_sentence_meteor(references, hypotheses)
        return {
            "BLEU-1": bleu1,
            "BLEU-2": bleu2,
            "BLEU-3": bleu3,
            "BLEU-4": bleu4,
            "ROUGE-L": rouge_l,
            "METEOR": meteor,
        }

    def test(self):
        self.model.eval()

        references = []
        hypotheses = []

        with torch.inference_mode():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = batch["image"]
                input_ids = batch["input_ids"]

                generated_ids = self.model.generate(
                    images,
                    max_length=self.eval_max_length,
                    num_beams=self.num_beams,
                )
                gt_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                references.extend(gt_texts)
                hypotheses.extend(pred_texts)

        results = self.evaluate_reports(references, hypotheses)
        print("Evaluation Metrics:")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medical", choices=["medical"])
    parser.add_argument("--data", type=str, default="/public/ATIQA/Datasets/iu_xray/")
    parser.add_argument("--image_root", type=str, default="/public/ATIQA/Datasets/iu_xray/")
    parser.add_argument("--model_dir", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--max_source_length", type=int, default=200)
    parser.add_argument("--eval_max_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--train_model_file", type=str, default="train_checkpoint_best.pt")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    train_loader, valid_loader, test_loader, num_classes = get_loaders_com(
        dataset="medical",
        args=args,
        tokenizer=tokenizer,
    )

    
    combined_model = EncoderDecoder(device=device, tokenizer=tokenizer).to(device)
     

    optimizer = optim.AdamW(
        combined_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    train(
        model=combined_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        device=device,
        train_model_file=args.train_model_file,
        
    )

    combined_model.load_state_dict(torch.load(args.train_model_file, map_location=device))
    print(f"Loaded best checkpoint from {args.train_model_file} for final evaluation")

    evaluator = TestEvaluator(
        model=combined_model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        device=device,
        eval_max_length=args.eval_max_length,
        num_beams=args.num_beams,
    )
    evaluator.test()

if __name__ == "__main__":

    main()
