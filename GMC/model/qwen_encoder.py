from transformers import AutoTokenizer, AutoModel 
import torch.nn as nn
import torch
from GMC.model.gcn_layer import GCNLayer
class QwenEncoder(nn.Module):
    def __init__(self, model_name="/public/model/Qwen2.5-1.5B-Instruct", proj_dim=512):
        super(QwenEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size

        self.mlp1 = nn.Linear(self.hidden_size, proj_dim)
        self.mlp2 = nn.Linear(self.hidden_size, proj_dim)
        self.mlp3 = nn.Linear(self.hidden_size, proj_dim)
        self.mlp4 = nn.Linear(proj_dim, self.hidden_size)

        self.text_gcn = GCNLayer(self.hidden_size, self.hidden_size)
        self.proj = nn.Linear(self.hidden_size, proj_dim)

        # Ensure pad_token is defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask, edge_index):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        B, T, D = token_embeddings.size()

        q = self.mlp1(token_embeddings)
        k = self.mlp2(token_embeddings)
        c = torch.bmm(q, k.transpose(1, 2)) / T
        c = torch.softmax(c, dim=-1)

        agg = torch.bmm(c, token_embeddings)
        out = self.mlp4(self.mlp3(agg))

        enhanced = out + token_embeddings  # Residual connection
        pooled = enhanced.mean(dim=1)
        return self.proj(pooled)
