import argparse
import sys
from pathlib import Path
import csv
import torch
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bertscore_score
BIOGPT_ROOT = Path('/public/ATIQA/BioGPTLLM')
sys.path.insert(0, str(BIOGPT_ROOT))
from src.dataset import get_loaders_com
from model_bridge import LatentMemoryBridgeModel


def clean_caption(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('|||', ' ')
    text = ' '.join(text.split())
    return text.strip()

def compute_bertscore_batch(ref_texts, hyp_texts, device):
    try:
        _, _, F1 = bertscore_score(
            cands=[clean_caption(x) for x in hyp_texts],
            refs=[clean_caption(x) for x in ref_texts],
            model_type="/public/ATIQA/multimodel_bridge/roberta-large",
            num_layers=17,
            lang=None,
            verbose=False,
            device=str(device)
        )
        return F1
    except Exception as e:
        print(f"BERTScore failed: {e}")
        return None
def compute_single_sample_scores(ref_text, hyp_text):
    gts = {0: [clean_caption(ref_text)]}
    res = {0: [clean_caption(hyp_text)]}

    sample_scores = {
        "BLEU-1": None,
        "BLEU-2": None,
        "BLEU-3": None,
        "BLEU-4": None,
        "METEOR": None,
        "ROUGE-L": None,
        "CIDEr": None
    }

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)

            if isinstance(method, list):
                for m, s in zip(method, score):
                    sample_scores[m] = float(s)
            else:
                if isinstance(scores, list):
                    sample_scores[method] = float(scores[0])
                else:
                    sample_scores[method] = float(score)

        except Exception as e:
            print(f"[Warning] {method} failed for one sample: {e}")

    return sample_scores


def save_sample_scores_to_csv(rows, csv_path):
    fieldnames = [
        "prediction",
        "gt",
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "meteor",
        "rougel",
        "cider",
        "bertscore_f1"
    ]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nPer-sample CSV saved to: {csv_path}")


def evaluate(model, loader, device,  max_new_tokens=56, num_beams=3, save_csv_path='test_results.csv'):
    model.eval()

    gts = {}
    res = {}
    sample_rows = []
    sample_id = 0
    all_ref_texts = []
    all_hyp_texts = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            images = batch['image'].to(device)
            ids = batch['input_ids']

            gen = model.generate(
                images,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )

            ref_texts = model.tokenizer.batch_decode(ids.cpu(), skip_special_tokens=True)
            hyp_texts = model.tokenizer.batch_decode(gen.cpu(), skip_special_tokens=True)

            for ref, hyp in zip(ref_texts, hyp_texts):
                ref_clean = clean_caption(ref)
                hyp_clean = clean_caption(hyp)
                all_ref_texts.append(ref_clean)
                all_hyp_texts.append(hyp_clean)

                gts[sample_id] = [ref_clean]
                res[sample_id] = [hyp_clean]
                

                one_scores = compute_single_sample_scores(ref_clean, hyp_clean)

                sample_rows.append({
                    "prediction": hyp_clean,
                    "gt": ref_clean,
                    "bleu1": one_scores["BLEU-1"],
                    "bleu2": one_scores["BLEU-2"],
                    "bleu3": one_scores["BLEU-3"],
                    "bleu4": one_scores["BLEU-4"],
                    "meteor": one_scores["METEOR"],
                    "rougel": one_scores["ROUGE-L"],
                    "cider": one_scores["CIDEr"]
                })

                sample_id += 1

  # ---- BERTScore for all pairs ----
    
    bert_f1 = compute_bertscore_batch(all_ref_texts, all_hyp_texts, device)

    if bert_f1 is not None:
        for i in range(len(sample_rows)):
            sample_rows[i]["bertscore_f1"] = float(bert_f1[i].item())
    else:
        for i in range(len(sample_rows)):
            sample_rows[i]["bertscore_f1"] = None

    save_sample_scores_to_csv(sample_rows, save_csv_path)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}

    for scorer, method in scorers:
        try:
            score,  = scorer.compute_score(gts, res)
        except Exception as e:
            print(f"{method} failed: {e}")
            if isinstance(method, list):
                for m in method:
                    final_scores[m] = None
            else:
                final_scores[method] = None
            continue

        if isinstance(method, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    if bert_f1 is not None:
        final_scores["BERTScore-F1"] = float(bert_f1.mean().item())
    else:
        final_scores["BERTScore-F1"] = None

    print("\n===== Average scores on full test set =====")
    for key in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore-F1"]:
        value = final_scores.get(key, None)
        if value is None:
            print(f"{key}: skipped")
        else:
            print(f"{key}: {value:.4f}")

    return final_scores, sample_rows
  

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--llama_ckpt', default='/public/ATIQA/multimodel_bridge/llama_stage2.pt')
    p.add_argument('--nvae_ckpt', default='/public/ATIQA/NVAEmaster/eval-exp/checkpoint.pt')
    p.add_argument('--data', default='/public/ATIQA/Datasets/iu_xray/')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_new_tokens', type=int, default=56)
    p.add_argument('--num_beams', type=int, default=4)
    p.add_argument('--text_model_name', default='/public/model/Llama-3.2-3B-Instruct')
    p.add_argument('--csv_path', type=str, default='/public/ATIQA/multimodel_bridge/test_results.csv')

    args = p.parse_args()

    args.nvae_ckpt = args.nvae_ckpt.strip()
    args.llama_ckpt = args.llama_ckpt.strip()
    args.csv_path = args.csv_path.strip()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.llama_ckpt, map_location='cpu')

    text_model_name = ck.get('text_model_name', args.text_model_name)
    mc_passes = ck.get('mc_passes', 10)

    model = LatentMemoryBridgeModel(
        nvae_ckpt=args.nvae_ckpt,
        text_model_name=text_model_name,
        mc_passes=mc_passes
    ).to(device)

    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    _, _, test_loader, _ = get_loaders_com(
        dataset='medical',
        args=argparse.Namespace(data=args.data, batch_size=args.batch_size),
        tokenizer=model.tokenizer,
    )

    evaluate(
        model=model,
        loader=test_loader,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        save_csv_path=args.csv_path
    )


if __name__ == '__main__':
    main()
    