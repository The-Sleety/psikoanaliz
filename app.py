import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
import textwrap

st.set_page_config(page_title="Psikolojik Görüntü Analizörü", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0a0a0f; color: #e8e6f0; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    
    /* Card-like containers */
    .metric-card {
        background-color: #12121f;
        border: 1px solid #22223a;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    
    .status-text { font-family: 'Courier New', Courier, monospace; color: #7c6af7; font-size: 0.9rem; }
    .trait-label { color: #e96dab; font-weight: bold; font-size: 1.5rem; letter-spacing: 1px; }
    .palette-box {
        height: 45px; width: 100%; border-radius: 6px; margin-bottom: 8px; 
        display: flex; align-items: center; justify-content: center;
        font-family: 'monospace'; font-size: 0.75rem; border: 1px solid #22223a;
    }
    hr { border-top: 1px solid #1a1a2e; }
    </style>
    """, unsafe_allow_html=True)

def _scale(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

def _hist_entropy(channel: np.ndarray) -> float:
    hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))

def analyze_image(img_pil: Image) -> dict:
    img_pil_small = img_pil.resize((256, 256), Image.LANCZOS)
    img = np.array(img_pil_small, dtype=np.float32)
    img_u8 = img.astype(np.uint8)

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (255 * edges.size)

    energy = _scale(0.45 * (val.mean()/255) + 0.30 * (gray.std()/128) + 0.25 * (sat.mean()/255), 0.0, 0.85)
    melancholy = _scale(0.35 * ((hue >= 90) & (hue <= 150)).astype(np.float32).mean() + 0.35 * (sat < 60).astype(np.float32).mean() + 0.30 * (1.0 - (val.mean()/255)), 0.0, 0.75)
    aggression = _scale(0.45 * np.clip((r - np.maximum(g, b)) / 255, 0, 1).mean() + 0.30 * edge_ratio + 0.25 * ((val < 60).astype(np.float32).mean()), 0.0, 0.65)
    serenity = _scale(0.35 * ((hue >= 60) & (hue <= 160)).astype(np.float32).mean() + 0.35 * (val.mean()/255) + 0.30 * (1.0 - edge_ratio), 0.0, 0.85)
    color_entr = (_hist_entropy(r) + _hist_entropy(g) + _hist_entropy(b)) / (3 * 8)
    complexity = _scale(0.50 * color_entr + 0.50 * edge_ratio, 0.0, 0.80)

    return {
        "Enerji": int(round(energy * 10)),
        "Melankoli": int(round(melancholy * 10)),
        "Saldırganlık": int(round(aggression * 10)),
        "Huzur": int(round(serenity * 10)),
        "Karmaşıklık": int(round(complexity * 10)),
        "meta": {
            "brightness": f"{val.mean()/2.55:.1f}%",
            "saturation": f"{sat.mean()/2.55:.1f}%",
            "contrast": f"{gray.std():.1f}",
            "edge_density": f"{edge_ratio*100:.2f}%"
        }
    }

def extract_palette(img_pil: Image, n: int = 6) -> list:
    img = np.array(img_pil.convert("RGB").resize((128, 128)))
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    return ["#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])) for c in centers[order]]



st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>◈ PSIKOLOJIK ANALIZ PANELI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b6882; margin-bottom: 30px;'>KURAL TABANLI GÖRÜNTÜ İŞLEME MOTORU v2.0</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Pikseller ayrıştırılıyor..."):
        res = analyze_image(image)
        scores = {k: v for k, v in res.items() if k != "meta"}
        meta = res["meta"]
        palette = extract_palette(image)

    col_img, col_pal, col_desc = st.columns([1.5, 0.8, 1.7])
    
    with col_img:
        st.image(image, use_container_width=True)
    
    with col_pal:
        st.markdown("### Renkler")
        for color in palette:
            st.markdown(f'<div class="palette-box" style="background-color:{color};">{color.upper()}</div>', unsafe_allow_html=True)

    with col_desc:
        top_trait = max(scores, key=scores.get)
        st.markdown(f"### Baskın Karakter")
        st.markdown(f"<div class='trait-label'>{top_trait.upper()}</div>", unsafe_allow_html=True)
        
        descriptions = {
            "Enerji": "Görsel, yüksek luminans ve doygunluk değerleri sayesinde izleyicide canlandırıcı bir etki bırakıyor. Dinamik bir kompozisyon yapısına sahip.",
            "Melankoli": "Düşük valörlü ve soğuk spektrumdaki renkler, içe dönük ve hüzünlü bir atmosfer yaratıyor. Sessizlik ve durgunluk hakim.",
            "Saldırganlık": "Keskin kenar geçişleri ve baskın sıcak tonlar, görselde bir gerilim ve iddialı bir duruş sergiliyor.",
            "Huzur": "Yumuşak kontrast geçişleri ve dengeli ışık dağılımı, zihinsel bir rahatlama ve sükunet hissi uyandırıyor.",
            "Karmaşıklık": "Yüksek detay yoğunluğu ve renk entropisi, görselin çözümlenmesi gereken zengin bir katmanlı yapıda olduğunu gösteriyor."
        }
        st.write(descriptions.get(top_trait))
        
        st.divider()
        st.markdown("### Teknik Veriler")
        c1, c2 = st.columns(2)
        c1.metric("Parlaklık", meta["brightness"])
        c1.metric("Doygunluk", meta["saturation"])
        c2.metric("Kontrast", meta["contrast"])
        c2.metric("Kenar Yoğ.", meta["edge_density"])

    st.markdown("---")
    vis_col1, vis_col2 = st.columns(2)

    categories = list(scores.keys())
    values = list(scores.values())

    with vis_col1:
        st.markdown("<h4 style='text-align: center;'>Boyutsal Dağılım</h4>", unsafe_allow_html=True)
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5), facecolor='#0a0a0f')
        ax_bar.set_facecolor('#12121f')
        y_pos = np.arange(len(categories))
        colors = ["#ff7eb3", "#7eb3ff", "#b3ffb3", "#ffb37e", "#d17eff"]
        
        ax_bar.barh(y_pos, values, color=colors, edgecolor="#22223a", height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(categories, color='#e8e6f0', fontsize=10)
        ax_bar.set_xlim(0, 10)
        ax_bar.xaxis.grid(True, color='#22223a', linestyle='--')
        ax_bar.set_axisbelow(True)
        
        for i, v in enumerate(values):
            ax_bar.text(v + 0.2, i, str(v), color='#e8e6f0', va='center', fontweight='bold')
        
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        
        st.pyplot(fig_bar)

    with vis_col2:
            st.markdown("<h4 style='text-align: center;'>Radar Profil Analizi</h4>", unsafe_allow_html=True)
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            vals_radar = values + values[:1]

            fig_radar = plt.figure(figsize=(6, 6), facecolor='#0a0a0f')
            ax_radar = fig_radar.add_subplot(111, polar=True)
            ax_radar.set_facecolor('#12121f')
            
            ax_radar.set_theta_offset(np.pi / 2)
            ax_radar.set_theta_direction(-1)
            
            ax_radar.plot(angles, vals_radar, color='#7c6af7', linewidth=2)
            ax_radar.fill(angles, vals_radar, color='#7c6af7', alpha=0.3)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, color='white', size=10, fontweight='bold')
            
            ax_radar.set_ylim(0, 10)
            ax_radar.set_yticks([2, 4, 6, 8, 10])
            ax_radar.set_yticklabels(["2", "4", "6", "8", "10"], color='white', size=8)
            
            ax_radar.grid(color="#33334d", linestyle='--')
            
            ax_radar.spines['polar'].set_color('#33334d')
            
            st.pyplot(fig_radar)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Analiz başlatmak için üstteki kutucuğa bir görsel sürükleyin.")
