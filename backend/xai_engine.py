# xai_engine.py
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
    'benefits_id': 'Benefit' 
}

def calculate_dynamic_stats(df: pd.DataFrame):
    stats = {}
    label_col = 'fraudulent' if 'fraudulent' in df.columns else ('label' if 'label' in df.columns else None)
    df_ref = df[df[label_col] == 0] if label_col else df
    if len(df_ref) == 0: df_ref = df

    for feat in FIELD_MAPPING.keys():
        if feat in df_ref.columns:
            lengths = df_ref[feat].astype(str).str.len()
            stats[feat] = {
                'mean': float(lengths.mean()) if not np.isnan(lengths.mean()) else 0.0,
                'std': float(lengths.std()) if not np.isnan(lengths.std()) or lengths.std() != 0 else 1.0
            }
        else:
            stats[feat] = {'mean': 0.0, 'std': 1.0}
    return stats


class XAIDiagnosticEngine:
    def __init__(self, field_statistics):
        self.stats = field_statistics

    def get_risk_label(self, val):
        abs_val = abs(val)
        if abs_val <= 0.02:   return "Sangat Rendah"
        elif abs_val <= 0.10: return "Rendah"
        elif abs_val <= 0.25: return "Sedang"
        else:                 return "Tinggi"

    def get_length_context(self, feat, current_len):
        stats = self.stats.get(feat)
        if not stats or stats['std'] == 0:
            return None  # Tidak ada anomali panjang

        z_score = (current_len - stats['mean']) / stats['std']
        if z_score < -1.96:
            return "terlalu singkat"
        elif z_score > 1.96:
            return "tidak wajar panjangnya"
        return None  # Normal, tidak perlu disebutkan

    def _length_note(self, length_flag, clean_name):
        if length_flag == "terlalu singkat":
            return f" Isi {clean_name} tergolong sangat singkat dibanding lowongan kerja pada umumnya."
        elif length_flag == "tidak wajar panjangnya":
            return f" Isi {clean_name} tergolong sangat panjang dan tidak lazim."
        return ""

    def analyze(self, row_cleaned_dict, current_shap, model, tokenizer, device):
        findings = []
        field_highlights = {}
        text_cols = list(FIELD_MAPPING.keys())

        # RULE: Banyak field kosong 
        empty_fields = [
            FIELD_MAPPING[feat] for feat in text_cols
            if str(row_cleaned_dict.get(feat, "")).strip().lower() in ['', 'nan', 'none']
        ]
        if len(empty_fields) >= 3:
            field_list = ", ".join(empty_fields)
            findings.append(
                f"Lowongan ini hanya mengisi sedikit informasi — sebanyak {len(empty_fields)} dari 5 bagian "
                f"tidak diisi ({field_list}). Lowongan kerja resmi umumnya mencantumkan informasi yang lengkap."
            )

        for feat in text_cols:
            idx = text_cols.index(feat)
            shap_val = current_shap[idx]
            clean_name = FIELD_MAPPING[feat]
            text_content = str(row_cleaned_dict.get(feat, "")).strip()
            risk_label = self.get_risk_label(shap_val)

            # 1. Field dikosongkan
            if text_content.lower() in ['', 'nan', 'none']:
                if shap_val > 0.02:
                    findings.append(
                        f"Bagian {clean_name} tidak diisi. "
                        f"Lowongan yang tidak mencantumkan {clean_name} cenderung lebih berisiko — "
                        f"tingkat kecurigaan: {risk_label}."
                    )
                continue

            text_len = len(text_content)
            length_flag = self.get_length_context(feat, text_len)
            length_note = self._length_note(length_flag, clean_name)

            # 2. Field mendorong prediksi fraud
            if shap_val > 0.05:
                findings.append(
                    f"Bagian {clean_name} terdeteksi memiliki pola yang mencurigakan.{length_note} "
                    f"Tingkat kecurigaan: {risk_label}."
                )
                try:
                    highlighted = get_model_based_highlights_html(model, tokenizer, row_cleaned_dict, feat, device)
                    if highlighted:
                        field_highlights[clean_name] = highlighted
                except Exception:
                    field_highlights[clean_name] = text_content

            # 3. Field mendukung keabsahan lowongan
            elif shap_val < -0.05:
                findings.append(
                    f"Bagian {clean_name} tampak wajar dan mendukung keabsahan lowongan ini. "
                    f"Tingkat kepercayaan: {risk_label}."
                )

        return findings, field_highlights


def get_model_based_highlights_html(model, tokenizer, full_input_dict, field_key, device, steps=24, threshold=0.04):
    target_text = full_input_dict.get(field_key, "")
    if not target_text or str(target_text).lower() in ['nan', 'none', '']: return ""

    model.eval()
    max_len = MAX_LENS[field_key]

    encoded = tokenizer(
        target_text, return_tensors='pt', truncation=True,
        max_length=max_len, padding='max_length', return_offsets_mapping=True
    )
    offsets = encoded.pop("offset_mapping")[0].numpy()

    input_ids      = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Baseline: embedding dari teks kosong ([CLS][SEP][PAD]...)
    encoded_baseline = tokenizer(
        "", return_tensors='pt', truncation=True,
        max_length=max_len, padding='max_length'
    ).to(device)
    baseline_ids = encoded_baseline['input_ids']

    with torch.no_grad():
        input_embeds    = model.bert.embeddings.word_embeddings(input_ids)
        baseline_embeds = model.bert.embeddings.word_embeddings(baseline_ids)

    # Static context dari field lain (dijaga konstan selama loop IG)
    def get_field_cls(f_k):
        t = str(full_input_dict.get(f_k, ""))
        e = tokenizer(
            t, max_length=MAX_LENS[f_k], padding='max_length',
            truncation=True, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            out = model.bert(
                input_ids=e['input_ids'],
                attention_mask=e['attention_mask']
            ).last_hidden_state[:, 0, :].unsqueeze(1)
        return out

    static_ctx = {f: get_field_cls(f) for f in MAX_LENS.keys() if f != field_key}

    alphas      = torch.linspace(0, 1, steps=steps).to(device)
    total_grads = torch.zeros_like(input_embeds).to(device)

    for alpha in alphas:
        interpolated = (
            baseline_embeds + alpha * (input_embeds - baseline_embeds)
        ).clone().detach().requires_grad_(True)

        target_output = model.bert(
            inputs_embeds=interpolated, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :].unsqueeze(1)

        ordered_outputs = []
        for f in ['title_id', 'company_profile_id', 'description_id', 'requirements_id', 'benefits_id']:
            ordered_outputs.append(target_output if f == field_key else static_ctx[f])

        field_embeddings = torch.cat(ordered_outputs, dim=1)
        attn_output, _   = model.attention_layer(field_embeddings, field_embeddings, field_embeddings)
        logits           = model.classifier_text_only(attn_output.reshape(attn_output.shape[0], -1))

        score = logits[0, 1]
        grads = torch.autograd.grad(outputs=score, inputs=interpolated)[0]
        total_grads += grads.detach()

    avg_grads      = total_grads / steps
    ig_attribution = ((input_embeds - baseline_embeds) * avg_grads).sum(dim=-1).squeeze(0).cpu().numpy()

    html_output, last_idx = "", 0
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue  # Lewati [CLS], [SEP], [PAD]

        token_score  = abs(ig_attribution[idx])
        html_output += target_text[last_idx:start]
        token_text   = target_text[start:end]

        if token_score > threshold:
            opacity      = min(token_score * 2, 0.85)
            html_output += (
                f"<span style='background-color: rgba(255,0,0,{opacity:.2f}); "
                f"padding:2px 3px; border-radius:4px; font-weight:bold;'>{token_text}</span>"
            )
        else:
            html_output += token_text

        last_idx = end

    html_output += target_text[last_idx:]
    return html_output