# model_loader.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MainModelTextOnly(nn.Module):
    def __init__(self, model_name):
        super(MainModelTextOnly, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Cross-Field Attention khusus untuk 5 field teks
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        
        self.dim_mode_text_only = (self.hidden_size * 5)

        # Classifier tunggal untuk skenario Text-Only (Dropout sesuai parameter latih = 0.2)
        self.classifier_text_only = nn.Sequential(
            nn.Linear(self.dim_mode_text_only, 512), 
            nn.LayerNorm(512), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(512, 2)
        )

    def forward(self, t_ids, t_mask, p_ids, p_mask, d_ids, d_mask, r_ids, r_mask, b_ids, b_mask):
        # Ekstraksi token [CLS] (indeks 0) dari masing-masing field teks
        o_t = self.bert(input_ids=t_ids, attention_mask=t_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_p = self.bert(input_ids=p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_d = self.bert(input_ids=d_ids, attention_mask=d_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_r = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        o_b = self.bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state[:, 0, :].unsqueeze(1)

        # Gabungkan ke dimensi (batch_size, 5, hidden_size)
        text_embeddings = torch.cat((o_t, o_p, o_d, o_r, o_b), dim=1)
        
        # Proses lewat Cross-Field Attention
        attn_output, _ = self.attention_layer(text_embeddings, text_embeddings, text_embeddings)
        text_features = attn_output.reshape(attn_output.shape[0], -1) 

        # Melewatkan output fitur ke classifier agar menghasilkan logits berdimensi 2 (Normal vs Fraud)
        return self.classifier_text_only(text_features)

def get_assets(path, model_name, device):
    model = MainModelTextOnly(model_name).to(device)
    
    # strict=False otomatis mengabaikan layer fusion metadata yang tidak dipakai di web app
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer