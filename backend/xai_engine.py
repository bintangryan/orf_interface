import torch
import numpy as np
import pandas as pd

MAX_LENS = {
    'title_id': 16, 
    'company_profile_id': 256, 
    'description_id': 512, 
    'requirements_id': 256, 
    'benefits_id': 150
}

FIELD_MAPPING = {
    'title_id': 'Posisi Pekerjaan',
    'company_profile_id': 'Profil Perusahaan',
    'description_id': 'Deskripsi Pekerjaan',
    'requirements_id': 'Persyaratan',
    'benefits_id': 'Benefits'
}

def calculate_dynamic_stats(df: pd.DataFrame):
    """
    Menghitung mean dan standard deviasi panjang karakter secara dinamis
    berdasarkan data lowongan kerja legitimate (label = 0 atau fraud = 0).
    """
    stats = {}
    
    # Prioritaskan memfilter data legitimate (0) agar menjadi acuan pembanding jomplang
    if 'label' in df.columns:
        df_ref = df[df['label'] == 0]
        if len(df_ref) == 0: df_ref = df
    elif 'fraud' in df.columns:
        df_ref = df[df['fraud'] == 0]
        if len(df_ref) == 0: df_ref = df
    else:
        df_ref = df

    for feat in FIELD_MAPPING.keys():
        if feat in df_ref.columns:
            lengths = df_ref[feat].astype(str).str.len()
            stats[feat] = {
                'mean': float(lengths.mean()) if not np.isnan(lengths.mean()) else 0.0,
                'std': float(lengths.std()) if not np.isnan(lengths.std()) else 1.0
            }
        else:
            stats[feat] = {'mean': 0.0, 'std': 1.0}
            
    return stats

class XAIDiagnosticEngine:
    def __init__(self, field_statistics):
        self.stats = field_statistics

    def get_influence_label(self, val):
        abs_val = abs(val)
        if abs_val <= 0.01: return "Sangat Rendah"
        elif abs_val <= 0.10: return "Rendah"
        elif abs_val <= 0.25: return "Sedang"
        else: return "Tinggi"

    def get_length_context(self, feat, current_len):
        """
        Menentukan konteks panjang teks menggunakan metode Z-score.
        """
        stats = self.stats.get(feat)
        if not stats or stats['std'] == 0:
            return "berada dalam rentang umum lowongan valid"

        # Rumus Z-score: (X - Mean) / Std_Dev
        z_score = (current_len - stats['mean']) / stats['std']

        # Batas ambang akademis (Threshold Z = ±1.96)
        if z_score < -1.96:
            return f"terdeteksi jauh lebih pendek dibanding sebagian besar lowongan valid pada data referensi"
        elif z_score > 1.96:
            return f"terdeteksi jauh lebih panjang dibanding sebagian besar lowongan valid pada data referensi"
        
        return f"berada dalam rentang panjang teks yang umum ditemukan pada data referensi lowongan valid"


def get_model_based_highlights_html(model, tokenizer, full_input_dict, field_key, device, steps=8, threshold=0.05):
    """
    Kalkulasi tingkat token dengan True Integrated Gradients (IG).
    """
    target_text = full_input_dict.get(field_key, "")
    if not target_text or str(target_text).lower() in ['nan', 'none', '']: 
        return ""
    
    model.eval()
    max_len = MAX_LENS[field_key]
    
    encoded = tokenizer(
        target_text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=max_len, 
        padding='max_length', 
        return_offsets_mapping=True
    )
    offset_mapping = encoded.pop("offset_mapping")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    input_embeds = model.bert.embeddings.word_embeddings(input_ids)
    baseline_embeds = torch.zeros_like(input_embeds).to(device)
    
    alphas = torch.linspace(0, 1, steps=steps).to(device)
    total_grads = torch.zeros_like(input_embeds).to(device)

    def get_real_context_embed(f_key):
        f_text = full_input_dict.get(f_key, "")
        txt = str(f_text) if f_text and str(f_text).lower() not in ['nan', 'none', ''] else ""
        
        enc_static = tokenizer(
            txt, 
            max_length=MAX_LENS[f_key], 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            out = model.bert(
                input_ids=enc_static['input_ids'], 
                attention_mask=enc_static['attention_mask']
            ).last_hidden_state[:, 0, :].unsqueeze(1)
        return out

    static_contexts = {}
    for f in ['title_id', 'company_profile_id', 'description_id', 'requirements_id', 'benefits_id']:
        if f != field_key:
            static_contexts[f] = get_real_context_embed(f)

    for alpha in alphas:
        interpolated = (baseline_embeds + alpha * (input_embeds - baseline_embeds)).detach().requires_grad_(True)
        model.zero_grad()
        
        target_output = model.bert(inputs_embeds=interpolated, attention_mask=attention_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        
        other_outputs = []
        for f in ['title_id', 'company_profile_id', 'description_id', 'requirements_id', 'benefits_id']:
            if f == field_key:
                other_outputs.append(target_output)
            else:
                other_outputs.append(static_contexts[f])
        
        field_embeddings = torch.cat(other_outputs, dim=1)
        attn_output, _ = model.global_attention(field_embeddings, field_embeddings, field_embeddings)
        logits = model.classifier(attn_output.reshape(attn_output.shape[0], -1))
        
        grads = torch.autograd.grad(outputs=logits[0, 1], inputs=interpolated)[0]
        total_grads += grads.detach()

    avg_grads = total_grads / steps
    ig_attr = ((input_embeds - baseline_embeds) * avg_grads).sum(dim=-1).squeeze(0).detach().cpu().numpy()
    offsets = offset_mapping[0].numpy()
    
    html_output, last_idx = "", 0
    for idx, (start, end) in enumerate(offsets):
        if start == end: continue
        
        token_score = abs(ig_attr[idx])
        html_output += target_text[last_idx:start]
        token_text = target_text[start:end]
        
        if token_score > threshold:
            opacity = min(token_score * 2, 0.85)
            html_output += f"<span style='background-color: rgba(255,0,0,{opacity}); padding:2px 3px; border-radius:4px; font-weight:bold;'>{token_text}</span>"
        else: 
            html_output += token_text
        last_idx = end
        
    html_output += target_text[last_idx:]
    return html_output