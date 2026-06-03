import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class GlobalAttentionHierarchy(nn.Module):
    def __init__(self, model_name):
        super(GlobalAttentionHierarchy, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.global_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        
        # DISESUAIKAN: Mengikuti arsitektur Model 1 agar weights bisa ter-load sempurna
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 512),
            nn.LayerNorm(512),  # <--- Tambahkan ini sesuai Model 1
            nn.ReLU(),
            nn.Dropout(0.3),    # <--- Naikkan ke 0.3 sesuai Model 1
            nn.Linear(512, 2)
        )

    def forward(self, t_ids, t_mask, p_ids, p_mask, d_ids, d_mask, r_ids, r_mask, b_ids, b_mask):
        o_t = self.bert(input_ids=t_ids, attention_mask=t_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_p = self.bert(input_ids=p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_d = self.bert(input_ids=d_ids, attention_mask=d_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_r = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_b = self.bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        
        field_embeddings = torch.cat((o_t, o_p, o_d, o_r, o_b), dim=1)
        attn_output, _ = self.global_attention(field_embeddings, field_embeddings, field_embeddings)
        combined = attn_output.reshape(attn_output.shape[0], -1) 
        return self.classifier(combined)

def get_assets(path, model_name, device):
    model = GlobalAttentionHierarchy(model_name).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer