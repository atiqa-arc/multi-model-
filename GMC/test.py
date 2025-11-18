import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import re
import gc
from decoder_train import FullDecoder 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 1️⃣ Load test data
# =====================================================
test_tokens_path = "/public/ATIQA/my_code/low_uncertainty_test_tokens.pt"
test_feats_path = "/public/ATIQA/my_code/low_uncertainty_test_features.pt"
test_csv_path = "/public/ATIQA/my_code/test_pairs.csv"

# Text variance embeddings
txt_embeddings = torch.load(test_tokens_path, map_location=device)
# Image certainty features
img_feats_list = torch.load(test_feats_path, map_location=device)
# Pad image features
max_img_len = max(t.shape[0] for t in img_feats_list)
hidden_dim = img_feats_list[0].shape[1]

padded_img_feats = [
    F.pad(img, (0, 0, 0, max_img_len - img.shape[0]), value=0.0)
    for img in img_feats_list
]

img_feats = torch.stack(padded_img_feats).to(device)
# Load ground-truth reports
df = pd.read_csv(test_csv_path)
gt_captions = df["captions"].tolist()
#assert len(gt_texts) == len(data), "Mismatch in dataset sizes!"

# =====================================================
# 2️⃣ Load tokenizer and model
# =====================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
VOCAB_SIZE = tokenizer.vocab_size
# IMPORT your FullDecoder class here
model = FullDecoder(
    hidden_dim=512,
    max_len=512,
    vocab_size=VOCAB_SIZE
).to(device)

model_path = "/public/ATIQA/my_code/best_decoder_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded!")

# =====================================================
# 3️⃣ Beam search decoding
# =====================================================

def beam_search(model, txt_emb, img_feat, beam_size=3, max_len=50):

    sequences = [[list(), 0.0]]  # ([tokens], score)

    for step in range(max_len):
        all_candidates = []

        for seq, score in sequences:
            # Prepare input
            if len(seq) == 0:
                prev_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
            else:
                prev_ids = torch.tensor([seq], device=device)

            # Run decoder
            logits, probs, _ = model(
                txt_embeddings=txt_emb.unsqueeze(0),
                img_features=img_feat.unsqueeze(0)
            )

            next_token_logits = logits[0, step]  # [vocab]
            prob = F.softmax(next_token_logits, dim=-1)
            top_k = torch.topk(prob, beam_size)

            for i in range(beam_size):
                next_token = top_k.indices[i].item()
                next_prob = top_k.values[i].item()

                candidate = [seq + [next_token], score - np.log(next_prob)]
                all_candidates.append(candidate)

        # Sort full list and keep best beams
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_size]

    best_seq = sequences[0][0]
    decoded = tokenizer.decode(best_seq, skip_special_tokens=True)
    return decoded


# =====================================================
# 4️⃣ Dirichlet uncertainty estimation
# =====================================================

def compute_dirichlet_uncertainty(logits):
    evidence = F.relu(logits)
    alpha = evidence + 1

    S = torch.sum(alpha, dim=-1)
    C = alpha.size(-1)

    uncertainty = C / S
    return uncertainty.detach().cpu().numpy()


def sentence_uncertainty(report, token_unc):
    sentences = re.split(r'[.!?]', report)
    sent_uncs = []

    idx = 0
    for s in sentences:
        length = len(tokenizer.encode(s))  # rough token count
        if length == 0:
            continue
        u = np.mean(token_unc[idx:idx+length])
        sent_uncs.append(u)
        idx += length

    report_unc = np.mean(sent_uncs) if len(sent_uncs) > 0 else 1.0
    return sent_uncs, report_unc


# =====================================================
# 5️⃣ Metrics (BLEU, METEOR, ROUGE)
# =====================================================

smooth = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_metrics(pred, ref):
    # Tokenize
    pred_tokens = pred.split()
    ref_tokens = ref.split()

    # BLEU
    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    # METEOR
    meteor = single_meteor_score(ref_tokens, pred_tokens)

    # ROUGE
    rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

    return bleu1, bleu2, bleu3, bleu4, meteor, rouge_l


# =====================================================
# 6️⃣ Run inference on test set
# =====================================================

predictions = []
sent_uncertainties = []
report_uncertainties = []
bleu1_list = []
bleu2_list = []
bleu3_list = []
bleu4_list = []
meteor_list = []
rouge_list = []

with torch.no_grad():     # <<— correct

    for i in range(len(txt_embeddings)):   # <<— inside "with"

        txt = txt_embeddings[i]
        img = img_feats[i]

        # ------------------------------------------
        # Decoding
        # ------------------------------------------
        pred = beam_search(model, txt, img, beam_size=3)
        gt = gt_captions[i]

        # ------------------------------------------
        # Dirichlet Uncertainty
        # ------------------------------------------
        logits, _, _ = model(
            txt_embeddings=txt.unsqueeze(0),
            img_features=img.unsqueeze(0)
        )

        token_u = compute_dirichlet_uncertainty(logits[0])
        su, ru = sentence_uncertainty(pred, token_u)

        # ------------------------------------------
        # Metrics
        # ------------------------------------------
        b1,b2,b3,b4,mtr,rl = compute_metrics(pred, gt)

        # ------------------------------------------
        # Save values
        # ------------------------------------------
        predictions.append(pred)
        sent_uncertainties.append(su)
        report_uncertainties.append(ru)
        bleu1_list.append(b1)
        bleu2_list.append(b2)
        bleu3_list.append(b3)
        bleu4_list.append(b4)
        meteor_list.append(mtr)
        rouge_list.append(rl)

        print(f"[{i}] Done.")

# =====================================================
# Save results
# =====================================================

out_df = pd.DataFrame({
    "prediction": predictions,
    "gt": gt_captions,
    "sentence_uncertainty": sent_uncertainties,
    "report_uncertainty": report_uncertainties,
    "bleu1": bleu1_list,
    "bleu2": bleu2_list,
    "bleu3": bleu3_list,
    "bleu4": bleu4_list,
    "meteor": meteor_list,
    "rougeL": rouge_list
})


out_df.to_csv("/public/ATIQA/my_code/test_results.csv", index=False)
print("Results saved to test_results.csv")
