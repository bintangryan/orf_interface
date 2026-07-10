"use client";

import { useState, useEffect, useRef } from "react";
import axios from "axios";

import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  Binary,
  Loader2,
  AlertCircle,
  Search,
  CheckCircle2,
  FileText,
  ArrowLeft,
  Sparkles,
  ChevronRight,
  ChevronDown,
} from "lucide-react";

type FieldKey = "title" | "profile" | "description" | "requirements" | "benefits";

interface ShapDataItem {
  label: string;
  value: number;
}

interface PredictionResult {
  prediction: "FRAUD" | "LEGIT";
  probability: number;
  base_value: number;
  shap_data: ShapDataItem[];
  findings: string[];
  highlights: Record<string, string>;
  empty_warning_msg?: string | null;
}

// ── Design tokens ──────────────────────────────────────────────────────────
const BLUE = "#3674B5";
const BLUE_DARK = "#2D629A";
const FRAUD_COLOR = "#BD114A";
const LEGIT_COLOR = "#1E7D6A";
const ease = "transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]";

const detectShapAnomalies = (shapData: ShapDataItem[]) => {
  if (!shapData || shapData.length < 2) return null;

  const absValues = shapData.map((item) => Math.abs(item.value));

  const mean = absValues.reduce((sum, v) => sum + v, 0) / absValues.length;
  const std = Math.sqrt(
    absValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / absValues.length
  );

  if (std < 0.01) return null;

  let dominantItem: ShapDataItem | null = null;
  let highestZ = 0;

  for (const item of shapData) {
    const absVal = Math.abs(item.value);
    const z = (absVal - mean) / std;

    if (z > 1.5 && absVal > mean && absVal > 0.08 && z > highestZ) {
      dominantItem = item;
      highestZ = z;
    }
  }

  if (!dominantItem) return null;

  return {
    label: dominantItem.label,
    value: dominantItem.value,
    zScore: highestZ,
    type: dominantItem.value < 0 ? "negatif" : "positif",
  };
};

export default function ORFPage() {
  const [started, setStarted] = useState(false);
  const [activeCard, setActiveCard] = useState<"fraud" | "valid">("fraud");
  const [activeTab, setActiveTab] = useState<FieldKey>("title");
  const [expandedShap, setExpandedShap] = useState<Record<string, boolean>>({});

  const [formData, setFormData] = useState<Record<FieldKey, string>>({
    title: "",
    profile: "",
    description: "",
    requirements: "",
    benefits: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  useEffect(() => {
    if (started) return;
    const interval = setInterval(() => {
      setActiveCard((prev) => (prev === "fraud" ? "valid" : "fraud"));
    }, 3500);
    return () => clearInterval(interval);
  }, [started]);

  useEffect(() => {
    if (scrollContainerRef.current && tabRefs.current[activeTab]) {
      const container = scrollContainerRef.current;
      const tab = tabRefs.current[activeTab];
      if (tab) {
        const scrollLeft = tab.offsetLeft - (container.clientWidth / 2) + (tab.clientWidth / 2);
        container.scrollTo({ left: scrollLeft, behavior: "smooth" });
      }
    }
  }, [activeTab]);

  const tabs: { id: FieldKey; label: string; placeholder: string; backendKey: string }[] = [
    { id: "title", label: "Posisi", placeholder: "Contoh: Staff Data Entry, Marketing Manager...", backendKey: "title_id" },
    { id: "profile", label: "Profil Perusahaan", placeholder: "Tempel deskripsi perusahaan di sini...", backendKey: "company_profile_id" },
    { id: "description", label: "Deskripsi Pekerjaan", placeholder: "Tanggung jawab dan detail pekerjaan...", backendKey: "description_id" },
    { id: "requirements", label: "Persyaratan", placeholder: "Kualifikasi dan kriteria kandidat...", backendKey: "requirements_id" },
    { id: "benefits", label: "Benefit", placeholder: "Gaji, tunjangan, dan fasilitas yang ditawarkan...", backendKey: "benefits_id" },
  ];

  const handleAnalyze = async () => {
    setLoading(true);
    setResult(null);
    setExpandedShap({});
    try {
      const payload = {
        title_id: formData.title,
        company_profile_id: formData.profile,
        description_id: formData.description,
        requirements_id: formData.requirements,
        benefits_id: formData.benefits,
      };
      
      // --- PERBAIKAN URL ---
      // Ambil URL dari environment variable (Vercel), jika tidak ada gunakan localhost
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
      const res = await axios.post(`${apiUrl}/predict`, payload);
      // ---------------------

      setResult(res.data);
    } catch (error) {
      console.error(error);
      alert("Koneksi ke backend gagal. Pastikan server berjalan di port 8000 atau URL sudah benar.");
    } finally {
      setLoading(false);
    }
  };

  const toggleExpandShap = (label: string) => {
    setExpandedShap((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  const filledCount = Object.values(formData).filter((v) => v.trim() !== "").length;
  const isFraud = result?.prediction === "FRAUD";
  const shapAnomaly = result ? detectShapAnomalies(result.shap_data) : null;

  if (!started) {
    const cardConfig = {
      fraud: {
        label: "FRAUD",
        pct: "94.1%",
        barW: "94%",
        color: FRAUD_COLOR,
        bg: "#FEF2F5",
        msg: "Model mendeteksi pola teks yang memiliki kemiripan dengan pola yang lebih sering muncul pada lowongan fraud di data pelatihan.",
        icon: <ShieldAlert size={20} strokeWidth={2.5} style={{ color: FRAUD_COLOR }} />,
        miniShap: [
          { label: "Deskripsi Pekerjaan", val: "+0.421", w: "70%", pos: true },
          { label: "Benefit", val: "+0.280", w: "45%", pos: true },
        ],
      },
      valid: {
        label: "VALID",
        pct: "18.4%",
        barW: "18%",
        color: LEGIT_COLOR,
        bg: "#EFFAF7",
        msg: "Model mendeteksi pola teks yang lebih dekat dengan karakteristik lowongan valid pada data pelatihan.",
        icon: <ShieldCheck size={20} strokeWidth={2.5} style={{ color: LEGIT_COLOR }} />,
        miniShap: [
          { label: "Profil Perusahaan", val: "-0.310", w: "50%", pos: false },
          { label: "Deskripsi Pekerjaan", val: "-0.150", w: "25%", pos: false },
        ],
      },
    };

    return (
      <main className="min-h-screen bg-slate-50 text-slate-800 font-sans">
        <div className="fixed inset-0 -z-10 pointer-events-none overflow-hidden">
          <div className="absolute top-0 right-0 w-[600px] h-[600px] rounded-full" style={{ background: `radial-gradient(circle, ${BLUE}08 0%, transparent 70%)` }} />
          <div className="absolute bottom-0 left-0 w-[500px] h-[500px] rounded-full" style={{ background: `radial-gradient(circle, ${FRAUD_COLOR}05 0%, transparent 70%)` }} />
        </div>

        <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-slate-200/60">
          <div className="max-w-6xl mx-auto px-6 lg:px-10 h-14 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center shadow-sm" style={{ backgroundColor: BLUE }}>
                <ShieldCheck size={14} className="text-white" strokeWidth={2.5} />
              </div>
              <span className="text-[16px] font-bold tracking-tight text-slate-900">
                ORF<span style={{ color: BLUE }}> Detection</span>
              </span>
            </div>
          </div>
        </nav>

        <section className="max-w-6xl mx-auto px-6 lg:px-10 pt-12 pb-20">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
            <div className="flex flex-col items-start">
              <h1 className="text-[32px] font-bold leading-[1.15] tracking-tight text-slate-900">
                Deteksi Lowongan Kerja Palsu<br />
                <span className="text-slate-400 font-semibold">dengan AI yang Transparan</span>
              </h1>
              <p className="mt-6 text-[16px] leading-relaxed text-slate-500 max-w-md">
                Didukung oleh Transformer IndoBERT dan Explainable AI, sistem siap membantu Anda mendeteksi lowongan kerja palsu sekaligus memaparkan alasan di baliknya
              </p>
              <button
                onClick={() => setStarted(true)}
                className={`group mt-10 inline-flex items-center gap-2.5 px-7 h-12 rounded-xl text-white text-[12px] font-bold shadow-md hover:shadow-lg ${ease} hover:-translate-y-px`}
                style={{ backgroundColor: BLUE }}
              >
                Mulai Analisis
                <ChevronRight size={18} className="group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>

            <div className="flex justify-center lg:justify-end mt-10 lg:mt-0">
              <div className="relative w-full max-w-[420px] h-[380px]">
                {(["valid", "fraud"] as const).map((type) => {
                  const cfg = cardConfig[type];
                  const isActive = activeCard === type;
                  const isFr = type === "fraud";
                  return (
                    <div
                      key={type}
                      onMouseEnter={() => setActiveCard(type)}
                      className={`absolute inset-0 rounded-3xl border bg-white p-7 cursor-default flex flex-col ${ease} ${
                        isActive
                          ? "z-20 scale-100 shadow-[0_12px_40px_-12px_rgba(0,0,0,0.15)] border-slate-200"
                          : `z-10 scale-95 opacity-60 shadow-sm border-slate-100 ${isFr ? "rotate-[3deg] translate-x-4 translate-y-5" : "-rotate-[3deg] translate-x-4 translate-y-5"}`
                      }`}
                    >
                      <div className="flex items-start justify-between mb-6">
                        <div>
                          <p className="text-[10px] uppercase tracking-[0.15em] text-slate-400 font-bold mb-1.5">Hasil Deteksi</p>
                          <div className="flex items-center gap-2.5">
                            <span className="text-[24px] font-black tracking-tight" style={{ color: cfg.color }}>{cfg.label}</span>
                            <span className="text-[12px] font-bold px-2 py-0.5 rounded-md" style={{ backgroundColor: cfg.bg, color: cfg.color }}>{cfg.pct}</span>
                          </div>
                        </div>
                        <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{ backgroundColor: cfg.bg }}>
                          {cfg.icon}
                        </div>
                      </div>

                      <div className="mb-6">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-[12px] text-slate-500 font-semibold">Probabilitas Fraud</span>
                          <span className="text-[12px] font-bold text-slate-700">{cfg.pct}</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${ease}`}
                            style={{ width: isActive ? cfg.barW : "0%", backgroundColor: cfg.color }}
                          />
                        </div>
                      </div>

                      <div className="rounded-xl p-4 border mb-6" style={{ backgroundColor: cfg.bg, borderColor: `${cfg.color}20` }}>
                        <p className="text-[12px] text-slate-700 leading-relaxed font-medium">{cfg.msg}</p>
                      </div>

                      <div className="mt-auto">
                        <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-3">Kontribusi Bagian Teks</p>
                        <div className="space-y-2.5">
                          {cfg.miniShap.map((shap, i) => (
                            <div key={i} className="flex items-center gap-3">
                              <span className="text-[10px] font-semibold text-slate-600 w-28 truncate">{shap.label}</span>
                              <div className="flex-1 h-1.5 rounded-full bg-slate-100 flex overflow-hidden">
                                <div className="basis-1/2 flex justify-end">
                                  {!shap.pos && <div className="h-full rounded-l-full" style={{ width: shap.w, backgroundColor: LEGIT_COLOR }} />}
                                </div>
                                <div className="w-[2px] h-full bg-slate-200 flex-shrink-0" />
                                <div className="basis-1/2 flex justify-start">
                                  {shap.pos && <div className="h-full rounded-r-full" style={{ width: shap.w, backgroundColor: FRAUD_COLOR }} />}
                                </div>
                              </div>
                              <span className="text-[10px] font-bold w-12 text-right" style={{ color: shap.pos ? FRAUD_COLOR : LEGIT_COLOR }}>
                                {shap.val}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="border-t border-slate-200/60 bg-white">
          <div className="max-w-6xl mx-auto px-6 lg:px-10 py-20">
            <div className="mb-10">
              <p className="text-[10px] uppercase tracking-[0.14em] text-slate-400 font-bold mb-2">Explainable AI Layer</p>
              <h2 className="text-[24px] font-bold tracking-tight text-slate-900">AI yang Tidak Hanya Menilai, Tetapi Juga Menjelaskan</h2>
              <p className="mt-2.5 text-[12px] text-slate-500 leading-relaxed max-w-lg">
                Sistem tidak hanya memberikan hasil deteksi, tetapi juga menunjukkan bagian teks, dan indikasi spesifik yang memengaruhi keputusan AI secara transparan
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-5">
              <div className="rounded-2xl border border-slate-200 bg-white p-7 hover:border-slate-300 hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-3 mb-7">
                  <div className="w-9 h-9 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
                    <Binary size={17} style={{ color: BLUE }} strokeWidth={2.5} />
                  </div>
                  <div>
                    <h3 className="text-[16px] font-bold text-slate-800">Risk Contribution Analysis</h3>
                    <p className="text-[12px] text-slate-400 font-medium">Menganalisis bagian yang memberikan kontribusi terbesar terhadap keputusan model</p>
                  </div>
                </div>
                <div className="space-y-5">
                  {[
                    { label: "Deskripsi Pekerjaan", value: "+0.421", w: "85%", pos: true },
                    { label: "Profil Perusahaan", value: "+0.214", w: "45%", pos: true },
                    { label: "Persyaratan", value: "-0.102", w: "20%", pos: false },
                  ].map((item, i) => (
                    <div key={i}>
                      <div className="flex items-baseline justify-between mb-2">
                        <span className="text-[12px] font-semibold text-slate-600">{item.label}</span>
                        <span className="text-[12px] font-bold tabular-nums" style={{ color: item.pos ? FRAUD_COLOR : LEGIT_COLOR }}>{item.value}</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-slate-100 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: item.w, backgroundColor: item.pos ? FRAUD_COLOR : LEGIT_COLOR }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-7 hover:border-slate-300 hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-3 mb-7">
                  <div className="w-9 h-9 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
                    <Zap size={17} style={{ color: BLUE }} strokeWidth={2.5} />
                  </div>
                  <div>
                    <h3 className="text-[16px] font-bold text-slate-800">Contextual Text Attribution</h3>
                    <p className="text-[12px] text-slate-400 font-medium">AI menandai kata atau token spesifik yang kontribusi gradiennya melampaui ambang batas sensitivitas model dalam mendorong hasil prediksi.</p>
                  </div>
                </div>
                <div className="rounded-xl border border-slate-100 bg-slate-50 p-5 text-[12px] leading-8 text-slate-600 font-medium">
                  Kami mencari kandidat untuk membantu{" "}
                  <mark className="px-1.5 py-0.5 rounded bg-red-50 text-red-700 font-bold border border-red-100 not-italic">
                    pengelolaan transaksi
                  </mark>{" "}
                  secara online dengan{" "}
                  <mark className="px-1.5 py-0.5 rounded bg-red-50 text-red-700 font-bold border border-red-100 not-italic">
                    bonus mingguan tinggi
                  </mark>{" "}
                  dan penggunaan{" "}
                  <mark className="px-1.5 py-0.5 rounded bg-red-50 text-red-700 font-bold border border-red-100 not-italic">
                    rekening pribadi
                  </mark>{" "}
                  untuk kebutuhan operasional.
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="h-screen flex flex-col bg-slate-100 overflow-hidden font-sans">
      <header className="flex-shrink-0 h-14 bg-white border-b border-slate-200/80 flex items-center justify-between px-6 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: BLUE }}>
            <ShieldCheck size={14} className="text-white" strokeWidth={2.5} />
          </div>
          <span className="text-[16px] font-bold tracking-tight text-slate-900">
            ORF<span style={{ color: BLUE }}> Detection</span>
          </span>
        </div>
        <button
          onClick={() => {
            setStarted(false);
            setResult(null);
            setFormData({ title: "", profile: "", description: "", requirements: "", benefits: "" });
          }}
          className="group flex items-center gap-1.5 text-[12px] font-semibold text-slate-400 hover:text-slate-700 transition-colors"
        >
          <ArrowLeft size={13} className="group-hover:-translate-x-0.5 transition-transform" />
          Beranda
        </button>
      </header>

      <div className="flex flex-1 min-h-0">
        <div className="w-full max-w-[400px] xl:max-w-[480px] flex-shrink-0 flex flex-col bg-white border-r border-slate-200/80 overflow-hidden">
          <div className="flex-shrink-0 px-5 py-4 border-b border-slate-100 bg-slate-50/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText size={16} className="text-slate-400" strokeWidth={2} />
                <span className="text-[14px] font-bold text-slate-700">Input Lowongan</span>
              </div>
              <div className="flex items-center gap-1.5">
                {tabs.map((t) => (
                  <div
                    key={t.id}
                    className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${
                      formData[t.id].trim() ? "bg-emerald-500" : "bg-slate-200"
                    }`}
                  />
                ))}
                <span className="text-[10px] font-bold text-slate-400 ml-1 tabular-nums">{filledCount}/5</span>
              </div>
            </div>
          </div>

          <div className="flex-shrink-0 px-4 pt-3 pb-3 border-b border-slate-100 bg-white">
            <div
              ref={scrollContainerRef}
              className="flex items-center gap-2 overflow-x-auto pb-1 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] scroll-smooth"
            >
              {tabs.map((tab) => {
                const isActive = activeTab === tab.id;
                const isFilled = formData[tab.id].trim() !== "";
                return (
                  <button
                    key={tab.id}
                    ref={(el) => { tabRefs.current[tab.id] = el; }}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-shrink-0 flex items-center gap-1.5 rounded-[10px] border ${ease} ${
                      isActive
                        ? "h-10 px-4 text-[10px] font-bold text-white"
                        : "h-8 px-3 text-[10px] font-semibold bg-white border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50 opacity-80 hover:opacity-100"
                    }`}
                    style={isActive ? { backgroundColor: BLUE, borderColor: BLUE } : {}}
                  >
                    <span className="whitespace-nowrap tracking-wide">{tab.label}</span>
                    {isFilled && (
                      <CheckCircle2
                        size={isActive ? 14 : 12}
                        className={`${ease} ${isActive ? "text-white/90" : "text-emerald-500/70"}`}
                        strokeWidth={isActive ? 2.5 : 2}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex-1 p-4 min-h-0 flex flex-col bg-slate-50/30">
            <div className="flex-1 flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm focus-within:border-blue-300 focus-within:ring-4 focus-within:ring-blue-50 transition-all duration-200 overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  {tabs.find((t) => t.id === activeTab)?.label}
                </span>
                {formData[activeTab].trim() !== "" && (
                  <span className="text-[10px] text-emerald-600 font-semibold bg-emerald-50 px-2 py-0.5 rounded-md">
                    Terisi
                  </span>
                )}
              </div>
              <textarea
                value={formData[activeTab]}
                onChange={(e) => setFormData({ ...formData, [activeTab]: e.target.value })}
                placeholder={tabs.find((t) => t.id === activeTab)?.placeholder}
                className="w-full flex-1 bg-transparent px-4 py-4 outline-none resize-none text-[12px] text-slate-700 placeholder:text-slate-400 leading-relaxed"
              />
            </div>
          </div>

          <div className="flex-shrink-0 p-4 border-t border-slate-100 bg-white">
            <button
              onClick={handleAnalyze}
              disabled={loading || filledCount === 0}
              className="w-full flex items-center justify-center gap-2 h-11 rounded-xl text-white text-[12px] font-bold disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 active:scale-[0.99] shadow-sm"
              style={{ backgroundColor: BLUE }}
            >
              {loading
                ? <><Loader2 size={16} className="animate-spin" strokeWidth={2.5} />Menganalisis Lowongan...</>
                : <><Sparkles size={16} strokeWidth={2.5} />Mulai Deteksi</>}
            </button>
            {filledCount === 0 && (
              <p className="text-center text-[10px] text-slate-400 mt-2.5 font-medium">Isi minimal 1 kolom untuk mulai deteksi</p>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
          {!result && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
              <div className="w-16 h-16 rounded-2xl bg-white border border-slate-200 flex items-center justify-center mb-5 shadow-sm">
                <Search size={24} className="text-slate-300" strokeWidth={2} />
              </div>
              <p className="text-[16px] font-bold text-slate-600 tracking-tight">Siap Melakukan Analisis</p>
              <p className="text-[12px] text-slate-400 mt-2 max-w-[280px] leading-relaxed">
                Input detail lowongan kerja pada panel yang tersedia untuk mulai deteksi
              </p>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center">
              <Loader2 size={32} strokeWidth={2} className="animate-spin mb-5" style={{ color: BLUE }} />
              <p className="text-[16px] font-bold text-slate-700">AI Memproses Lowongan...</p>
              <p className="text-[12px] text-slate-400 mt-1.5 font-medium">Menelusuri pola teks dan menghitung kontribusi setiap bagian terhadap hasil deteksi</p>
              <div className="mt-8 flex gap-1.5">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="w-1.5 h-6 rounded-full animate-pulse"
                    style={{ backgroundColor: `${BLUE}40`, animationDelay: `${i * 100}ms` }}
                  />
                ))}
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="absolute inset-0 flex flex-col bg-slate-50 overflow-y-auto">
              
              <div className={`sticky top-0 z-10 flex-shrink-0 px-6 lg:px-8 py-4 bg-white flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b shadow-md ${
                isFraud ? "border-red-600/30 shadow-red-600/5" : "border-emerald-600/30 shadow-emerald-600/5"
              }`}>
                <div className="flex items-center gap-4">
                  <div className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0" style={{ backgroundColor: isFraud ? "#FEF2F5" : "#EFFAF7" }}>
                    {isFraud
                      ? <ShieldAlert size={20} strokeWidth={2.5} style={{ color: FRAUD_COLOR }} />
                      : <ShieldCheck size={20} strokeWidth={2.5} style={{ color: LEGIT_COLOR }} />}
                  </div>
                  <div>
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="text-[24px] font-bold tracking-tight" style={{ color: isFraud ? FRAUD_COLOR : LEGIT_COLOR }}>
                        {result.prediction === "LEGIT" ? "VALID" : result.prediction}
                      </span>
                      <span
                        className="text-[12px] font-bold px-2.5 py-0.5 rounded-md border"
                        style={{
                          color: isFraud ? FRAUD_COLOR : LEGIT_COLOR,
                          backgroundColor: isFraud ? "#FEF2F5" : "#EFFAF7",
                          borderColor: isFraud ? `${FRAUD_COLOR}25` : `${LEGIT_COLOR}25`,
                        }}
                      >
                        {(result.probability * 100).toFixed(1)}% Probabilitas Fraud
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="max-w-4xl mx-auto w-full p-4 lg:p-6 space-y-6">
                
                <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-sm">
                  <p className="text-[13px] font-bold text-slate-400 uppercase tracking-widest mb-4">Tingkat Risiko Lowongan</p>
                  <div className="flex items-baseline gap-1.5 mb-4">
                    <span className="text-[36px] font-black leading-none tabular-nums tracking-tight" style={{ color: isFraud ? FRAUD_COLOR : LEGIT_COLOR }}>
                      {(result.probability * 100).toFixed(1)}
                    </span>
                    <span className="text-[24px] font-bold text-slate-300">%</span>
                    <span
                      className="text-[12px] font-bold ml-3 px-2.5 py-0.5 rounded-lg border"
                      style={{ 
                        backgroundColor: isFraud ? "#FEF2F5" : "#EFFAF7", 
                        color: isFraud ? FRAUD_COLOR : LEGIT_COLOR,
                        borderColor: isFraud ? `${FRAUD_COLOR}15` : `${LEGIT_COLOR}15`
                      }}
                    >
                      {isFraud ? "Risiko Tinggi (Palsu)" : "Risiko Rendah (Valid)"}
                    </span>
                  </div>
                  
                  <div className="relative w-full h-3 rounded-full bg-slate-100 overflow-hidden mb-2">
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out"
                      style={{ width: `${(result.probability * 100).toFixed(1)}%`, backgroundColor: isFraud ? FRAUD_COLOR : LEGIT_COLOR }}
                    />
                  </div>
                  <div className="flex justify-between text-[11px] font-bold uppercase tracking-widest text-slate-400">
                    <span>0% — Valid</span>
                    <span>100% — Fraud</span>
                  </div>
                </div>

                {shapAnomaly && (
                  <div className="bg-amber-50 rounded-2xl border border-amber-200/80 p-5 shadow-sm flex items-start gap-4 animate-fade-in">
                    <div className="w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center flex-shrink-0">
                      <AlertCircle size={20} className="text-amber-600" strokeWidth={2.5} />
                    </div>

                    <div className="flex-1">
                      <h4 className="text-[14px] font-bold text-amber-900 tracking-tight">
                        Perhatian: Terdeteksi Komponen dengan Pengaruh Dominan
                      </h4>

                      <p className="text-[12px] text-amber-700 mt-1 leading-relaxed font-medium">
                        Komponen <span className="font-bold">"{shapAnomaly.label}"</span> teridentifikasi sebagai bagian yang memiliki pengaruh paling dominan dibandingkan komponen lainnya dalam hasil analisis model.
                      </p>

                      <p className="text-[10px] text-amber-600/90 mt-2.5 leading-relaxed">
                        Karena memberikan kontribusi yang relatif dominan terhadap hasil analisis model, 
                        bagian ini disarankan untuk ditinjau kembali secara manual. 
                      </p>
                    </div>
                  </div>
                )}

                {result.empty_warning_msg && (
                  <div className="bg-red-50 rounded-2xl border border-red-200/80 p-5 shadow-sm flex items-start gap-4 animate-fade-in">
                    <div className="w-10 h-10 rounded-xl bg-red-100 flex items-center justify-center flex-shrink-0">
                      <AlertCircle size={20} className="text-red-600" strokeWidth={2.5} />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-[14px] font-bold text-red-900 tracking-tight">
                        Peringatan: Informasi Tidak Lengkap
                      </h4>
                      <p className="text-[12px] text-red-700 mt-1 leading-relaxed font-medium">
                        {result.empty_warning_msg}
                      </p>
                    </div>
                  </div>
                )}

                <div className="bg-white rounded-2xl border border-slate-200/80 p-5 lg:p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-100">
                    <div className="w-9 h-9 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
                      <Binary size={18} style={{ color: BLUE }} strokeWidth={2.5} />
                    </div>
                    <div>
                      <h3 className="text-[15px] font-bold text-slate-800">Analisis Pengaruh Setiap Komponen Lowongan</h3>
                      <p className="text-[12px] text-slate-400 mt-0.5">Klik bar kontribusi di bawah untuk melihat temuan khusus dan analisis interpretasi.</p>
                    </div>
                  </div>

                  <div className="mb-6 flex flex-wrap items-center gap-6 bg-slate-50/80 p-3 rounded-xl border border-slate-100">
                    <div className="flex items-center gap-2">
                      <div className="w-3.5 h-2 rounded-full" style={{ backgroundColor: FRAUD_COLOR }} />
                      <span className="text-[11px] font-bold text-slate-500">Meningkatkan risiko fraud</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3.5 h-2 rounded-full" style={{ backgroundColor: LEGIT_COLOR }} />
                      <span className="text-[11px] font-bold text-slate-500">Menurunkan risiko fraud (Valid)</span>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {result.shap_data?.map((item, idx) => {
                      const isPos = item.value > 0;
                      const isExpanded = !!expandedShap[item.label];

                      const barWidth = Math.min((Math.abs(item.value) / 0.6) * 100, 100);

                      let fieldKey: FieldKey | null = null;
                      const labelLower = item.label.toLowerCase();
                      
                      if (labelLower.includes("posisi") || labelLower.includes("pekerjaan") || labelLower.includes("title")) fieldKey = "title";
                      else if (labelLower.includes("profil") || labelLower.includes("perusahaan") || labelLower.includes("profile")) fieldKey = "profile";
                      else if (labelLower.includes("deskripsi") || labelLower.includes("description")) fieldKey = "description";
                      else if (labelLower.includes("persyaratan") || labelLower.includes("requirements")) fieldKey = "requirements";
                      else if (labelLower.includes("benefit") || labelLower.includes("keuntungan") || labelLower.includes("benefits")) fieldKey = "benefits";

                      const localHighlightHtml = result.highlights ? result.highlights[item.label] : "";

                      const matchedFindings = result.findings?.filter(f => {
                        return f.toLowerCase().includes(item.label.toLowerCase()) || (fieldKey && f.toLowerCase().includes(fieldKey));
                      }) || [];

                      const showTextHighlighting = item.value > 0.05;
                      const isLowInfluence = item.value >= -0.05 && item.value <= 0.05;

                      const leftBarColor = LEGIT_COLOR;
                      const rightBarColor = FRAUD_COLOR;

                      return (
                        <div 
                          key={idx} 
                          className={`border rounded-xl transition-all duration-200 overflow-hidden ${
                            isExpanded ? "border-slate-300 bg-slate-50/40 shadow-sm" : "border-slate-100 bg-white hover:border-slate-200"
                          }`}
                        >
                          <div 
                            onClick={() => toggleExpandShap(item.label)}
                            className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 cursor-pointer select-none hover:bg-slate-50/80 transition-colors group"
                          >
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <span className="text-[13px] font-bold text-slate-700">{item.label}</span>
                                  {matchedFindings.length > 0 && (
                                    <span className="text-[10px] text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded font-semibold">
                                      {matchedFindings.length} Analisis
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="text-[12px] font-bold tabular-nums" style={{ color: isPos ? FRAUD_COLOR : LEGIT_COLOR }}>
                                    {item.value > 0 ? "+" : ""}{item.value.toFixed(3)}
                                  </span>
                                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 w-14 text-right">
                                    {isPos ? "↑ fraud" : "↓ valid"}
                                  </span>
                                </div>
                              </div>

                              <div className="w-full h-2.5 rounded-full bg-slate-100 flex overflow-hidden">
                                <div className="basis-1/2 flex justify-end">
                                  {!isPos && <div className="h-full rounded-l-full" style={{ width: `${barWidth}%`, backgroundColor: leftBarColor }} />}
                                </div>
                                <div className="w-0.5 h-full bg-slate-300 z-10 flex-shrink-0" />
                                <div className="basis-1/2 flex justify-start">
                                  {isPos && <div className="h-full rounded-r-full" style={{ width: `${barWidth}%`, backgroundColor: rightBarColor }} />}
                                </div>
                              </div>
                            </div>

                            <div 
                              className={`flex items-center justify-center gap-1.5 w-28 h-8 rounded-lg text-[11px] font-bold border transition-all hover:opacity-90 ${
                                isExpanded 
                                  ? "bg-slate-100 border-slate-200 text-slate-600" // Warna saat kebuka (Tutup)
                                  : "text-white shadow-sm" // Warna saat tertutup (Lihat Analisis)
                              }`}
                              style={!isExpanded ? { backgroundColor: BLUE, borderColor: BLUE } : {}}
                            >
                              <span>{isExpanded ? "Tutup" : "Lihat Analisis"}</span>
                              <ChevronDown 
                                size={14} 
                                className={`transition-transform duration-300 ${isExpanded ? "rotate-180" : ""}`} 
                              />
                            </div>
                          </div>

                          {isExpanded && (
                            <div className="px-4 pb-4 pt-1 border-t border-slate-100 bg-white space-y-4">
                              
                              {matchedFindings.length > 0 ? (
                                <div className="space-y-2 pt-2">
                                  <div className="flex items-center gap-1.5 text-slate-500">
                                    <AlertCircle size={13} className="text-amber-500" />
                                    <span className="text-[11px] font-bold uppercase tracking-wider">TEMUAN</span>
                                  </div>
                                  <div className="space-y-2">
                                    {matchedFindings.map((finding, fIdx) => (
                                      <div key={fIdx} className="p-3 bg-amber-50/40 border border-amber-100 rounded-lg text-[12px] text-slate-700 leading-relaxed font-medium">
                                        {finding}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}

                              {isLowInfluence && (
                                <div className="pt-2">
                                  <div className="p-3 bg-slate-50 border border-slate-200/60 rounded-lg text-[12px] text-slate-500 leading-relaxed font-medium flex items-start gap-2">
                                    <AlertCircle size={15} className="text-slate-400 flex-shrink-0 mt-0.5" />
                                    <span>
                                      Tidak ada temuan pendukung pada field ini. Nilai kontribusi fitur berada dalam rentang ambang batas netral model AI.
                                    </span>
                                  </div>
                                </div>
                              )}

                              {showTextHighlighting && localHighlightHtml && (
                                <div className="space-y-2 pt-1">
                                  <div className="flex items-center gap-1.5 text-slate-500">
                                    <FileText size={13} className="text-blue-500" />
                                    <span className="text-[11px] font-bold uppercase tracking-wider">Text Highlighting</span>
                                  </div>
                                  <div 
                                    className="p-4 bg-slate-50 border border-slate-100 rounded-xl text-[12px] font-medium leading-7 text-slate-700 break-words"
                                    dangerouslySetInnerHTML={{ __html: localHighlightHtml }}
                                  />
                                </div>
                              )}

                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {result.findings && result.findings.filter(f => f.includes("Distribusi") || f.includes("seimbang")).map((f, i) => (
                  <div key={i} className="bg-emerald-50 rounded-2xl border border-emerald-200 p-5 shadow-sm flex items-start gap-4 animate-fade-in">
                    <div className="w-10 h-10 rounded-xl bg-emerald-100 flex items-center justify-center flex-shrink-0">
                      <ShieldCheck size={20} style={{ color: LEGIT_COLOR }} strokeWidth={2.5} />
                    </div>
                    <div>
                      <h4 className="text-[14px] font-bold text-emerald-900 tracking-tight">Kepadatan Informasi Terverifikasi Aman</h4>
                      <p className="text-[12px] text-emerald-700 mt-1 leading-relaxed font-medium">{f}</p>
                    </div>
                  </div>
                ))}

              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}