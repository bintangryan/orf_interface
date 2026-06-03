from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import shap
import numpy as np
import re
from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware
from model_loader import get_assets
from xai_engine import XAIDiagnosticEngine, get_model_based_highlights_html, FIELD_MAPPING, MAX_LENS, calculate_dynamic_stats

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "C:\\its\\TUGAS AKHIR\\INTERFACE\\orf-system\\MODEL_BY_FIELD\\model_IndoBERT_TextOnly_S1_20.pth"
MODEL_NAME = "indobenchmark/indobert-base-p2"
CSV_PATH = "C:\\its\\TUGAS AKHIR\\INTERFACE\\orf-system\\models\\train_40.csv"

# ── FUNGSI CLEANING TEKS DARI PROSES TRAINING KAGGLE/COLAB ──
def clean_text(text):
    if pd.isna(text) or text == "" or not isinstance(text, str):
        return ""

    # 1. Hilangkan tag HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Spasi antara CamelCase dan tanda baca
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,!?:])([a-zA-Z0-9])', r'\1 \2', text)

    # 3. Bersihkan token URL/ID panjang khas dataset
    text = re.sub(r'#?URL[_\s]*[a-zA-Z0-9]{15,}#?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[a-zA-Z0-9]{15,}', ' ', text)

    # 4. Hapus URL standar dan Email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)

    # 5. Buang special character tersisa (termasuk bullet point •) & bersihkan spasi
    text = re.sub(r"[^a-zA-Z0-9\s.,!?:']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

# Load Assets
print(f"--- RUNNING ON DEVICE: {DEVICE} ---")
model, tokenizer = get_assets(MODEL_PATH, MODEL_NAME, DEVICE)
df_test = pd.read_csv(CSV_PATH)

print("--- CALCULATING DYNAMIC FIELD STATISTICS FROM CSV ---")
FIELD_STATS = calculate_dynamic_stats(df_test)
print("Dynamic Stats Result:", FIELD_STATS)

text_cols = list(FIELD_MAPPING.keys())

# Ambil background data dan bersihkan dengan clean_text agar sejalan saat kalkulasi SHAP
print("--- PREPROCESSING BACKGROUND DATA FOR SHAP ---")
bg_data = df_test[text_cols].sample(64, random_state=42).copy()
for col in text_cols:
    bg_data[col] = bg_data[col].astype(str).apply(clean_text)

def predict_fn(input_data):
    if isinstance(input_data, np.ndarray):
        input_df = pd.DataFrame(input_data, columns=text_cols)
    else:
        input_df = input_data.copy()
    
    batch_size = 48
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_df), batch_size):
            batch_df = input_df.iloc[i : i + batch_size]
            
            def enc(texts, key):
                return tokenizer(
                    texts.astype(str).tolist(), 
                    max_length=MAX_LENS[key], 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(DEVICE)

            t = enc(batch_df['title_id'], 'title_id')
            p = enc(batch_df['company_profile_id'], 'company_profile_id')
            d = enc(batch_df['description_id'], 'description_id')
            r = enc(batch_df['requirements_id'], 'requirements_id')
            b = enc(batch_df['benefits_id'], 'benefits_id')

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(
                    t['input_ids'], t['attention_mask'], 
                    p['input_ids'], p['attention_mask'], 
                    d['input_ids'], d['attention_mask'], 
                    r['input_ids'], r['attention_mask'], 
                    b['input_ids'], b['attention_mask']
                )
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
            
            del t, p, d, r, b, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.array(all_probs)

print("--- INITIALIZING SHAP EXPLAINER ---")
explainer = shap.KernelExplainer(predict_fn, bg_data)

class JobInput(BaseModel):
    title_id: str
    company_profile_id: str
    description_id: str
    requirements_id: str
    benefits_id: str

@app.post("/predict")
async def predict(data: JobInput):
    try:
        # 1. Jalankan proses cleaning data utama
        cleaned_list = [
            clean_text(data.title_id),
            clean_text(data.company_profile_id),
            clean_text(data.description_id),
            clean_text(data.requirements_id),
            clean_text(data.benefits_id)
        ]
        cleaned_row = pd.DataFrame([cleaned_list], columns=text_cols)
        
        # Buat dictionary murni versi CLEANED untuk kebutuhan Integrated Gradients (IG)
        cleaned_job_dict = {
            'title_id': cleaned_list[0],
            'company_profile_id': cleaned_list[1],
            'description_id': cleaned_list[2],
            'requirements_id': cleaned_list[3],
            'benefits_id': cleaned_list[4]
        }
        
        # Prediction & SHAP menggunakan data cleaned
        prob = float(predict_fn(cleaned_row)[0])
        shap_values = explainer.shap_values(cleaned_row, nsamples=32).flatten()
        
        # 2. Diagnostic Engine
        engine = XAIDiagnosticEngine(FIELD_STATS)
        findings = []
        highlights = {}
        legit_like_fields = 0
        
        for i, feat in enumerate(text_cols):
            shap_val = float(shap_values[i])
            text_cleaned_content = str(cleaned_list[i])
            clean_name = FIELD_MAPPING[feat]
            influence = engine.get_influence_label(shap_val)
            
            # [LOGIC: EMPTY FIELD]
            if text_cleaned_content.lower() in ['', 'nan', 'none']:
                if shap_val > 0.02:
                    findings.append(f"Bagian '{clean_name}' tidak memiliki konten. Dalam prediksi model, kondisi ini teridentifikasi berkontribusi terhadap peningkatan probabilitas fraud (Pengaruh {influence}).")
                continue

            # [LOGIC: POSITIVE CONTRIBUTION]
            if shap_val > 0.05:
                context = engine.get_length_context(feat, len(text_cleaned_content))
                findings.append(f"Model mendeteksi bagian '{clean_name}' berkontribusi terhadap peningkatan probabilitas fraud dengan pengaruh {influence}. Panjang teks {context}.")
                
                # ── [PERBAIKAN] SAFE FALLBACK HANDLING UNTUK INTEGRATED GRADIENTS ──
                try:
                    highlights_html = get_model_based_highlights_html(model, tokenizer, cleaned_job_dict, feat, DEVICE)
                    # Jika xai_engine gagal mencocokkan string atau return kosong, fallback ke teks bersih biasa
                    highlights[feat] = highlights_html if highlights_html else text_cleaned_content
                except Exception as e:
                    print(f"⚠️ Gagal memproses Integrated Gradients pada field {feat}: {e}")
                    highlights[feat] = text_cleaned_content
            
            # [LOGIC: NEGATIVE CONTRIBUTION]
            elif shap_val < -0.05:
                legit_like_fields += 1
                findings.append(f"Model mendeteksi bagian '{clean_name}' berkontribusi dalam menurunkan probabilitas fraud dengan pengaruh {influence}. Pola teks pada field ini memiliki kemiripan dengan pola yang lebih sering muncul pada lowongan legitimate di data pelatihan.")
                if feat in ['description_id', 'requirements_id']:
                    findings.append(f"Konten pada field '{clean_name}' terdeteksi memiliki pola teks yang dalam proses pelatihan model lebih sering diasosiasikan dengan lowongan legitimate.")

        if legit_like_fields >= 3:
            findings.append("Beberapa field teridentifikasi memberikan kontribusi dalam menurunkan probabilitas fraud, dengan pola yang relatif konsisten terhadap lowongan legitimate pada data pelatihan model.")

        return {
            "prediction": "FRAUD" if prob > 0.5 else "LEGIT",
            "probability": prob,
            "base_value": float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
            "shap_data": [{"label": FIELD_MAPPING[f], "value": float(shap_values[i])} for i, f in enumerate(text_cols)],
            "findings": findings,
            "highlights": highlights
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)