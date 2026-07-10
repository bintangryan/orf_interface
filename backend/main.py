# main.py
import os
import torch
import pandas as pd
import shap
import numpy as np
import re
import contextlib
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model_loader import get_assets
from xai_engine import XAIDiagnosticEngine, FIELD_MAPPING, calculate_dynamic_stats

# --- KONFIGURASI PATH DINAMIS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "dataset", "train_20.csv")
TEST_CSV_PATH  = os.path.join(BASE_DIR, "dataset", "test_20f.csv")
MODEL_NAME = "indobenchmark/indobert-base-p2"
# --------------------------------

app = FastAPI(title="Online Recruitment Fraud Diagnostic System API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_cols = list(FIELD_MAPPING.keys())

# --- DEKLARASI GLOBAL VARIABLE ---
# Agar aset AI tidak diload berkali-kali setiap ada request prediksi
model = None
tokenizer = None
explainer = None
FIELD_STATS = None

def clean_text(text):
    if pd.isna(text) or text == "" or not isinstance(text, str): return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,!?:])([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?:']", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def predict_fn(input_data):
    if isinstance(input_data, np.ndarray):
        input_df = pd.DataFrame(input_data, columns=text_cols).astype(str)
    else:
        input_df = input_data.copy().astype(str)

    all_probs = []
    model.eval()

    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else contextlib.nullcontext():
            for i in range(0, len(input_df), 32): 
                batch_df = input_df.iloc[i : i + 32]

                def enc(texts_series, max_l):
                    texts_list = [str(x) for x in texts_series.cpu().values] if hasattr(texts_series, 'cpu') else [str(x) for x in texts_series.tolist()]
                    return tokenizer(
                        texts_list, max_length=max_l, padding='max_length', truncation=True, return_tensors='pt'
                    ).to(DEVICE)

                t = enc(batch_df['title_id'],          16)
                p = enc(batch_df['company_profile_id'], 256)
                d = enc(batch_df['description_id'],     512)
                r = enc(batch_df['requirements_id'],    256)
                b = enc(batch_df['benefits_id'],        150)

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

    return np.array(all_probs)

# --- STARTUP EVENT (RAHASIANYA ADA DI SINI) ---
@app.on_event("startup")
def load_ai_assets():
    global model, tokenizer, explainer, FIELD_STATS
    print("Server FastAPI telah siap dan membuka port. Memuat aset AI di background...")
    
    model, tokenizer = get_assets(MODEL_PATH, MODEL_NAME, DEVICE)
    
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_train_clean = df_train.copy()
    for col in text_cols:
        if col in df_train_clean.columns:
            df_train_clean[col] = df_train_clean[col].astype(str).apply(clean_text)

    FIELD_STATS = calculate_dynamic_stats(df_train_clean)

    label_col  = 'fraudulent' if 'fraudulent' in df_train.columns else 'label'
    df_train_legit = df_train[df_train[label_col] == 0]

    bg_data = df_train_legit[text_cols].sample(32, random_state=42).copy().reset_index(drop=True)
    for col in text_cols:
        bg_data[col] = bg_data[col].astype(str).apply(clean_text)

    print("Menginisialisasi SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(predict_fn, bg_data)
    print("Sistem Deteksi ORF sepenuhnya siap digunakan!")

class JobInput(BaseModel):
    title_id:           str
    company_profile_id: str
    description_id:     str
    requirements_id:    str
    benefits_id:        str

@app.post("/predict")
def predict(data: JobInput):
    # Proteksi: Cegah akses jika model belum selesai di-load saat server baru menyala
    if explainer is None or model is None:
        raise HTTPException(status_code=503, detail="Sistem AI masih dalam proses pemuatan awal. Mohon tunggu beberapa detik dan coba lagi.")
        
    try:
        cleaned_list = [
            clean_text(data.title_id),
            clean_text(data.company_profile_id),
            clean_text(data.description_id),
            clean_text(data.requirements_id),
            clean_text(data.benefits_id),
        ]
        cleaned_row     = pd.DataFrame([cleaned_list], columns=text_cols)
        cleaned_job_dict = {text_cols[i]: cleaned_list[i] for i in range(len(text_cols))}

        prob = float(predict_fn(cleaned_row)[0])
        raw_shap   = explainer.shap_values(cleaned_row, nsamples=100)
        shap_values = raw_shap[1].flatten() if isinstance(raw_shap, list) else raw_shap.flatten()

        engine = XAIDiagnosticEngine(FIELD_STATS)
        findings, highlights, empty_warning_msg = engine.analyze(
            row_cleaned_dict=cleaned_job_dict,
            current_shap=shap_values,
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )

        for feat in text_cols:
            clean_name = FIELD_MAPPING[feat]
            if clean_name not in highlights and cleaned_job_dict[feat] != "":
                highlights[clean_name] = getattr(data, feat)

        ev = explainer.expected_value
        base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

        return {
            "prediction":  "FRAUD" if prob > 0.5 else "LEGIT",
            "probability": prob,
            "base_value":  base_value,
            "shap_data":   [
                {"label": FIELD_MAPPING[f], "value": float(shap_values[i])}
                for i, f in enumerate(text_cols)
            ],
            "findings":    findings,
            "highlights":  highlights,
            "empty_warning_msg": empty_warning_msg,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
