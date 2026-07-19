# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px

# Import our custom preprocessing module
import preprocessing

# Page configuration
st.set_page_config(
    page_title="Fake Review Detector Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply Inter font universally */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Title typography styling */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Modern, solid flat button styling */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
        height: 3em !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.2) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.3) !important;
    }
    
    /* Premium bordered cards for prediction results */
    .prediction-card-genuine {
        background-color: #064e3b !important;
        border: 1px solid #059669 !important;
        border-left: 6px solid #10b981 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .prediction-card-fake {
        background-color: #7f1d1d !important;
        border: 1px solid #dc2626 !important;
        border-left: 6px solid #ef4444 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Metric Containers styling */
    [data-testid="metric-container"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        padding: 16px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Average metrics from the training dataset for comparison
dataset_averages = {
    'text_length': {'Genuine': 266.54, 'Fake': 252.36},
    'exclamation_count': {'Genuine': 0.40, 'Fake': 0.25},
    'uppercase_ratio': {'Genuine': 0.0118, 'Fake': 0.0083},
    'sentiment_score': {'Genuine': 0.5321, 'Fake': 0.6008},
    'lexical_diversity': {'Genuine': 0.8559, 'Fake': 0.7819}
}

# Helper to analyze pattern / subtype of fake review
def get_fake_subclasses(text, sentiment_score, lexical_diversity):
    promo_words = {'discount', 'promo', 'sponsored', 'affiliate', 'buy', 'deal',
                   'recommend', 'price', 'coupon', 'sale', 'offer', 'free', 'click',
                   'shipping', 'code', 'purchase', 'review', 'product', 'promote'}
    words = text.lower().split()
    word_count = len(words)
    
    subclasses = []
    if abs(sentiment_score) > 0.75 and word_count > 20:
        subclasses.append("Deceptive (Ulasan Manipulatif / Rekayasa)")
    if lexical_diversity < 0.40:
        subclasses.append("Bot-Generated (Dibuat Generator Otomatis / Bot)")
    if word_count < 15:
        subclasses.append("Spam (Teks Terlalu Singkat)")
    if sum(1 for w in words if w in promo_words) >= 2:
        subclasses.append("Paid/Promotional (Ulasan Iklan / Berbayar / Promosi)")
        
    return subclasses if subclasses else ["Unclassified (Palsu Tanpa Pola Khusus)"]

# Helper to run Pure Python SVM Prediction
def predict_review(text, weights):
    # 0. Automatic Translation to English
    translated_text = text
    was_translated = False
    try:
        from deep_translator import GoogleTranslator
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        if translated_text.strip().lower() != text.strip().lower():
            was_translated = True
    except Exception:
        pass

    # 1. Preprocess text
    clean_text = preprocessing.preprocess_text(translated_text)
    
    # 2. Tokenize (unigram + bigram)
    words = clean_text.split()
    unigrams = [w for w in words if len(w) > 0]
    bigrams = [f"{unigrams[i]} {unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    tokens = unigrams + bigrams
    
    # 3. Term frequency counts
    token_counts = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1
        
    # 4. Build TF-IDF vector
    vocab = weights['vocabulary']
    idf = weights['idf']
    coef = weights['coef']
    intercept = weights['intercept']
    
    n_features = len(vocab)
    tfidf_vec = np.zeros(n_features)
    
    for term, count in token_counts.items():
        if term in vocab:
            idx = vocab[term]
            tfidf_vec[idx] = count * idf[idx]
            
    # L2 Normalization
    norm = np.linalg.norm(tfidf_vec)
    if norm > 0:
        tfidf_vec = tfidf_vec / norm
        
    # 5. Extract and Scale linguistic features
    fe_df = preprocessing.extract_features(translated_text)
    scaler_mean = np.array(weights['scaler_mean'])
    scaler_scale = np.array(weights['scaler_scale'])
    
    fe_vals = fe_df.iloc[0].values
    scaled_fe = (fe_vals - scaler_mean) / scaler_scale
    
    # 6. Combine features
    X_combined = np.concatenate([tfidf_vec, scaled_fe])
    
    # 7. Predict using SVM decision boundary
    dec_val = np.dot(X_combined, coef) + intercept
    prediction = 1 if dec_val > 0 else 0
    confidence = (1 / (1 + np.exp(-abs(dec_val)))) * 100
    
    return prediction, dec_val, confidence, translated_text, was_translated, clean_text, fe_df

# Helper to generate XAI word highlighting
def highlight_words_html(translated_text, vocab, coef):
    highlighted_html = []
    words = translated_text.split()
    
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
    except Exception:
        stemmer = None
        
    for word in words:
        clean_w = word.lower().strip(".,!?;:()\"'")
        
        # Stem word to match vectorizer vocabulary keys
        if stemmer and clean_w:
            stemmed_w = stemmer.stem(clean_w)
        else:
            stemmed_w = clean_w
            
        if stemmed_w in vocab:
            idx = vocab[stemmed_w]
            weight = coef[idx]
            
            # Opacity depends on coefficient magnitude
            if weight > 0.02:  # Leaning towards Fake (Red)
                opacity = min(0.1 + abs(weight) * 0.8, 0.4)
                highlighted_html.append(f'<span style="background-color: rgba(239, 68, 68, {opacity}); border-bottom: 2px solid #ef4444; padding: 2px 4px; border-radius: 4px; color: inherit;">{word}</span>')
            elif weight < -0.02:  # Leaning towards Genuine (Green)
                opacity = min(0.1 + abs(weight) * 0.8, 0.4)
                highlighted_html.append(f'<span style="background-color: rgba(16, 185, 129, {opacity}); border-bottom: 2px solid #10b981; padding: 2px 4px; border-radius: 4px; color: inherit;">{word}</span>')
            else:
                highlighted_html.append(word)
        else:
            highlighted_html.append(word)
            
    return " ".join(highlighted_html)

# Sidebar Content
st.sidebar.title("Menu Utama")
st.sidebar.markdown("""
Aplikasi hibrida **Fake Review Detector** menggunakan model **Linear Support Vector Machine (Linear SVM)** yang dilatih pada *Mexwell Fake Reviews Dataset* (~40.094 ulasan).
""")

st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navigasi Dasbor",
    [
        "Demo Deteksi Ulasan", 
        "Dashboard EDA", 
        "Evaluasi & Performa Model", 
        "Metodologi & Dokumentasi"
    ]
)

# Top-level Title
st.title("Sistem Analisis & Deteksi Fake Review E-Commerce")
st.markdown("Deteksi keaslian ulasan produk secara akurat menggunakan gabungan representasi kata (TF-IDF) dan gaya penulisan stilometrik-sentimen.")
st.write("---")

# ==============================================================================
# HALAMAN 1: DEMO DETEKSI ULASAN
# ==============================================================================
if menu == "Demo Deteksi Ulasan":
    st.subheader("Demo Klasifikasi Keaslian Ulasan")
    
    weights_path = 'outputs/results/model_weights.json'
    
    # Verify weights JSON exists
    if not os.path.exists(weights_path):
        st.error("Error: Berkas outputs/results/model_weights.json tidak ditemukan. Silakan jalankan ekspor parameter di notebook terlebih dahulu.")
        st.stop()
        
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights = json.load(f)
        
    st.markdown("#### Uji Model Secara Real-Time")
    st.write("Anda bisa mengetik ulasan secara manual atau memilih contoh yang disediakan di bawah ini:")
    
    examples = {
        "Pilih Contoh Ulasan...": "",
        "Review Asli (Genuine Example 1)": "Love this movie and got it at a great prize.  Will recommend it to a friend.  Happy with the purchase.",
        "Review Asli (Genuine Example 2)": "This was one of the best mc reads and I read at least 6 novels a week was always on edge not knowing what's around the corner... Definite recommend reads can't wait for another",
        "Review Palsu (Fake Example 1)": "The book has it all: a sweet, sweet, romantic romance, a happy ending, and a HEA. I had a hard time putting it down!",
        "Review Palsu (Fake Example 2)": "I absolutely love this silk pillow.  It's comfortable and it's lovely.  I'm very pleased with the quality of the pillow.  The pillow itself is a great quality and"
    }
    
    selected_example = st.selectbox("Pilih salah satu contoh ulasan:", list(examples.keys()))
    
    default_text = ""
    if selected_example != "Pilih Contoh Ulasan...":
        default_text = examples[selected_example]
        
    col1, col2 = st.columns([3, 2])
    
    with col1:
        user_input = st.text_area(
            "Masukkan teks ulasan produk (Bahasa Indonesia / Inggris):",
            value=default_text,
            height=180,
            placeholder="Ketik ulasan Anda di sini..."
        )
        analyze_btn = st.button("Mulai Analisis Keaslian")
        
    with col2:
        st.markdown("""
        **Petunjuk Singkat Analisis:**
        *   **Ulasan Asli (Genuine)**: Biasanya menceritakan detail spesifik pengalaman pemakaian produk dengan variasi kata yang kaya dan ekspresi emosi sentimen yang seimbang.
        *   **Ulasan Palsu (Fake)**: Biasanya dibuat komputer/penulis bayaran. Cenderung berulang-ulang, berisi kalimat iklan, atau sentimen yang terlalu ekstrem.
        *   *Dukungan Bahasa*: Anda dapat mengetik dalam Bahasa Indonesia, sistem akan menerjemahkannya secara otomatis ke Bahasa Inggris di latar belakang untuk klasifikasi.
        """)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu sebelum menekan tombol analisis.")
        else:
            with st.spinner("Menerjemahkan dan menganalisis teks ulasan..."):
                # Run Pure Python inference pipeline
                prediction, dec_val, confidence, translated_text, was_translated, clean_text, fe_df = predict_review(user_input, weights)
                
                st.write("---")
                st.subheader("Hasil Analisis Prediksi")
                
                if was_translated:
                    st.info(f"Terjemahan Bahasa Inggris: *\"{translated_text}\"*")
                
                input_vals = fe_df.iloc[0].to_dict()

                # Display Prediction Card
                if prediction == 0:
                    st.markdown(f"""
                    <div class="prediction-card-genuine">
                        <h3 style="color: #34d399; margin:0;">GENUINE (Ulasan Asli)</h3>
                        <p style="margin: 5px 0 0 0; color: #f1f5f9;">
                            Model memprediksi ulasan ini adalah <b>ASLI (Original)</b> yang ditulis oleh pengguna nyata.<br>
                            Tingkat Keyakinan Model: <b>{confidence:.2f}%</b> (Jarak ke batas keputusan: {dec_val:.3f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Calculate specific fake review pattern subclasses
                    subclasses = get_fake_subclasses(translated_text, input_vals['sentiment_score'], input_vals['lexical_diversity'])
                    subclasses_str = ", ".join(subclasses)
                    st.markdown(f"""
                    <div class="prediction-card-fake">
                        <h3 style="color: #f87171; margin:0;">FAKE (Ulasan Palsu)</h3>
                        <p style="margin: 5px 0 0 0; color: #f1f5f9;">
                            Model memprediksi ulasan ini adalah <b>PALSU / DIBUAT KOMPUTER (Computer-Generated)</b>.<br>
                            Kategori Pola Palsu Terdeteksi: <b style="color: #fca5a5;">{subclasses_str}</b><br>
                            Tingkat Keyakinan Model: <b>{confidence:.2f}%</b> (Jarak ke batas keputusan: {dec_val:.3f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Real-time XAI Word Highlighting Section
                st.markdown("#### Interpretasi Kata (Explainable AI)")
                st.write("Kata-kata berikut diwarnai berdasarkan kontribusinya terhadap keputusan model (Merah: Cenderung Palsu, Hijau: Cenderung Asli):")
                
                highlighted_html = highlight_words_html(translated_text, weights['vocabulary'], weights['coef'])
                st.markdown(f'<div style="line-height: 1.8; background-color: #0f172a; border: 1px solid #1e293b; padding: 16px; border-radius: 8px; font-size: 1.1em;">{highlighted_html}</div>', unsafe_allow_html=True)
                st.write("")
                
                # Metrics comparison section
                st.subheader("Perbandingan Karakteristik Linguistik (Gaya Bahasa)")
                
                col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
                
                col_metric1.metric("Panjang Karakter", f"{input_vals['text_length']:.0f}",
                                   f"{input_vals['text_length'] - dataset_averages['text_length']['Genuine']:.1f} vs Rerata Asli")
                
                col_metric2.metric("Jumlah Tanda Seru", f"{input_vals['exclamation_count']:.0f}",
                                   f"{input_vals['exclamation_count'] - dataset_averages['exclamation_count']['Genuine']:.2f} vs Rerata Asli")
                
                col_metric3.metric("Rasio Huruf Kapital", f"{input_vals['uppercase_ratio']:.4%}",
                                   f"{input_vals['uppercase_ratio'] - dataset_averages['uppercase_ratio']['Genuine']:.4%} vs Rerata Asli")
                
                col_metric4.metric("Skor Sentimen VADER", f"{input_vals['sentiment_score']:.4f}",
                                   f"{input_vals['sentiment_score'] - dataset_averages['sentiment_score']['Genuine']:.4f} vs Rerata Asli")
                
                col_metric5.metric("Keragaman Kosakata", f"{input_vals['lexical_diversity']:.4f}",
                                   f"{input_vals['lexical_diversity'] - dataset_averages['lexical_diversity']['Genuine']:.4f} vs Rerata Asli")
                
                comp_data = {
                    'Karakteristik Teks': [
                        'Panjang Karakter (Total)', 
                        'Jumlah Tanda Seru (!)', 
                        'Rasio Kata Kapital Penuh (ALL-CAPS)', 
                        'Skor Sentimen VADER (Negatif ke Positif)', 
                        'Keragaman Kosakata (TTR)'
                    ],
                    'Ulasan Ini': [
                        f"{input_vals['text_length']:.0f}",
                        f"{input_vals['exclamation_count']:.0f}",
                        f"{input_vals['uppercase_ratio']:.4%}",
                        f"{input_vals['sentiment_score']:.4f}",
                        f"{input_vals['lexical_diversity']:.4f}"
                    ],
                    'Rerata Ulasan Asli (Genuine)': [
                        f"{dataset_averages['text_length']['Genuine']:.2f}",
                        f"{dataset_averages['exclamation_count']['Genuine']:.2f}",
                        f"{dataset_averages['uppercase_ratio']['Genuine']:.4%}",
                        f"{dataset_averages['sentiment_score']['Genuine']:.4f}",
                        f"{dataset_averages['lexical_diversity']['Genuine']:.4f}"
                    ],
                    'Rerata Ulasan Palsu (Fake)': [
                        f"{dataset_averages['text_length']['Fake']:.2f}",
                        f"{dataset_averages['exclamation_count']['Fake']:.2f}",
                        f"{dataset_averages['uppercase_ratio']['Fake']:.4%}",
                        f"{dataset_averages['sentiment_score']['Fake']:.4f}",
                        f"{dataset_averages['lexical_diversity']['Fake']:.4f}"
                    ]
                }
                
                st.table(pd.DataFrame(comp_data))
                
                st.subheader("Analisis Pola Kebahasaan Terdeteksi")
                patterns = []
                
                promo_words = {'discount', 'promo', 'sponsored', 'affiliate', 'buy', 'deal',
                               'recommend', 'price', 'coupon', 'sale', 'offer', 'free', 'click',
                               'shipping', 'code', 'purchase', 'review', 'product', 'promote'}
                words = translated_text.lower().split()
                found_promo = [w for w in words if w in promo_words]
                
                if len(found_promo) >= 2:
                    patterns.append(f"**Pola Promosi/Iklan Terdeteksi**: Mengandung kata-kata penjualan: `{list(set(found_promo))}`.")
                    
                if input_vals['lexical_diversity'] < 0.70:
                    patterns.append("**Pola Kosakata Repetitif (Keragaman Rendah)**: Ulasan menggunakan kata-kata yang diulang secara monoton, yang sering kali merupakan indikasi ulasan otomatis (bot-generated).")
                    
                if abs(input_vals['sentiment_score']) > 0.85:
                    patterns.append(f"**Pola Sentimen Ekstrem ({'Sangat Positif' if input_vals['sentiment_score'] > 0 else 'Sangat Negatif'})**: Ulasan mengandung bobot emosi yang sangat tinggi (skor: {input_vals['sentiment_score']:.2f}), yang sering digunakan oleh pembuat ulasan palsu untuk memengaruhi penilaian produk.")
                    
                if input_vals['text_length'] < 100:
                    patterns.append("**Ulasan Terlalu Singkat**: Panjang teks di bawah rata-rata umum, sehingga kurang memberikan ulasan deskriptif yang mendalam.")
                    
                if not patterns:
                    st.success("Gaya penulisan ulasan ini terlihat wajar dan natural, tanpa pola promosi mencurigakan atau pengulangan berlebih.")
                else:
                    for pat in patterns:
                        st.info(pat)

# ==============================================================================
# TAB 2: DASHBOARD EDA
# ==============================================================================
elif menu == "Dashboard EDA":
    st.subheader("Dashboard Exploratory Data Analysis (EDA)")
    st.write("Visualisasi berikut merepresentasikan karakteristik umum dataset ulasan e-commerce Mexwell yang digunakan dalam riset ini.")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.markdown("### 1. Distribusi Kelas Ulasan (Asli vs Palsu)")
        st.write("Dataset ini memiliki distribusi kelas yang sangat seimbang, sehingga sangat ideal untuk pelatihan tanpa bias kelas mayoritas.")
        dist_df = pd.DataFrame({
            'Label': ['Genuine (Original)', 'Fake (Computer-Generated)'],
            'Jumlah': [20164, 19930]
        })
        fig1 = px.pie(dist_df, values='Jumlah', names='Label', 
                      color='Label', color_discrete_map={'Genuine (Original)': '#10b981', 'Fake (Computer-Generated)': '#ef4444'},
                      hole=0.4, template='plotly_dark')
        fig1.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)
            
    with col_eda2:
        st.markdown("### 2. Hubungan Korelasi Fitur Stilometrik-Sentimen")
        st.write("Heatmap di bawah membuktikan bahwa korelasi antar fitur linguistik yang diekstrak mendekati nol (ketiadaan multikolinearitas).")
        corr_path = 'outputs/results/feature_correlation.csv'
        if os.path.exists(corr_path):
            corr_df = pd.read_csv(corr_path, index_col=0)
            fig2 = px.imshow(corr_df, text_auto=".4f", aspect="auto", 
                             color_continuous_scale='RdBu', origin='lower',
                             template='plotly_dark')
            fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Berkas feature_correlation.csv tidak ditemukan.")
            
    st.write("---")
    
    col_eda3, col_eda4 = st.columns(2)
    
    with col_eda3:
        st.markdown("### 3. Analisis Kata Kunci Dominan (N-Gram)")
        st.write("Visualisasi unigram dan bigram teratas menunjukkan ulasan palsu didominasi kosa kata iklan/promosi, sementara ulasan asli didominasi oleh kata-kata pengalaman emosi natural.")
        
        ngram_type = st.radio("Pilih Tingkat Kata (N-Gram):", ["Unigram (Satu Kata)", "Bigram (Dua Kata)"], key="eda_ngram_radio")
        if ngram_type == "Unigram (Satu Kata)":
            gen_data = pd.DataFrame({
                'Kata': ['one', 'great', 'book', 'like', 'good', 'well', 'love', 'would', 'really', 'get'][::-1],
                'Frekuensi': [4795, 4724, 3784, 3713, 3564, 3233, 3163, 2934, 2579, 2414][::-1]
            })
            fake_data = pd.DataFrame({
                'Kata': ['great', 'good', 'book', 'love', 'one', 'well', 'story', 'would', 'read', 'little'][::-1],
                'Frekuensi': [9758, 7834, 7463, 5984, 5969, 5384, 5197, 4832, 4569, 4529][::-1]
            })
        else:
            gen_data = pd.DataFrame({
                'Kata': ['year old', 'well made', 'works great', 'highly recommend', 'would recommend', 'good quality', 'works well', 'really like', 'well written', 'read book'][::-1],
                'Frekuensi': [526, 346, 299, 283, 254, 201, 199, 198, 187, 182][::-1]
            })
            fake_data = pd.DataFrame({
                'Kata': ['would recommend', 'well developed', 'characters well', 'well written', 'recommend anyone', 'highly recommend', 'works great', 'story well', 'well made', 'read book'][::-1],
                'Frekuensi': [1976, 1036, 1012, 969, 913, 886, 832, 758, 748, 680][::-1]
            })
            
        col_ng1, col_ng2 = st.columns(2)
        with col_ng1:
            fig_gen = px.bar(gen_data, x='Frekuensi', y='Kata', orientation='h', 
                             title='Genuine Reviews', color_discrete_sequence=['#10b981'],
                             template='plotly_dark')
            fig_gen.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_gen, use_container_width=True)
        with col_ng2:
            fig_fake = px.bar(fake_data, x='Frekuensi', y='Kata', orientation='h', 
                              title='Fake Reviews', color_discrete_sequence=['#ef4444'],
                              template='plotly_dark')
            fig_fake.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_fake, use_container_width=True)
            
    with col_eda4:
        st.markdown("### 4. Distribusi Sebaran Pola Heuristik Kebahasaan")
        st.write("Menampilkan persentase ulasan berdasarkan karakteristik aturan bahasa tertentu seperti sentimen ekstrem, keanekaragaman kosakata rendah, ulasan sangat pendek, dan promosi.")
        pattern_df = pd.DataFrame({
            'Pola Heuristik': ['Deceptive', 'Unclassified', 'Spam', 'Paid/Promotional', 'Bot-Generated'],
            'Jumlah': [1768, 1616, 496, 300, 64]
        })
        fig4 = px.bar(pattern_df, x='Jumlah', y='Pola Heuristik', orientation='h',
                      color='Pola Heuristik', color_discrete_sequence=px.colors.qualitative.Pastel1,
                      template='plotly_dark')
        fig4.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ==============================================================================
# TAB 3: EVALUASI & PERFORMA MODEL
# ==============================================================================
elif menu == "Evaluasi & Performa Model":
    st.subheader("Hasil Evaluasi Komprehensif Model")
    st.write("Berikut adalah perbandingan performa dari 4 jenis algoritma klasifikasi pada 2 skenario eksperimen.")
    
    # Load and show Experiment results from CSV
    st.markdown("### Perbandingan Performa Eksperimen")
    results_path = 'outputs/results/exp2_feature_engineering.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'F1-Score'], color='#d4efdf'))
        
        # Plotly Grouped Bar Chart for Model Comparison
        melted_df = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                    var_name='Metrik', value_name='Nilai')
        fig_comp = px.bar(melted_df, x='Model', y='Nilai', color='Metrik', barmode='group',
                           color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                           template='plotly_dark')
        fig_comp.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, yaxis_range=[0.6, 1.0])
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Berkas hasil eksperimen exp2_feature_engineering.csv tidak ditemukan.")
        
    st.write("---")
    
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.markdown("### Confusion Matrix (Model SVM Terbaik)")
        st.write("Matriks Kebingungan pada data uji untuk model terpilih (Linear SVM + Feature Engineering) menunjukkan tingkat klasifikasi benar yang sangat tinggi pada kedua kelas.")
        
        # Dynamic Plotly Confusion Matrix
        cm = np.array([[3584, 449], [516, 3470]])
        fig_cm = px.imshow(cm, text_auto=True, 
                          x=['Predicted Genuine', 'Predicted Fake'],
                          y=['Actual Genuine', 'Actual Fake'],
                          color_continuous_scale='Blues',
                          template='plotly_dark')
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
            
    with col_eval2:
        st.markdown("### Stabilitas Performa 5-Fold Cross-Validation")
        st.write("Akurasi dan kestabilan generalisasi model diukur secara internal menggunakan 5-Fold Stratified Cross-Validation pada training set.")
        cv_path = 'outputs/results/cross_validation_results.csv'
        if os.path.exists(cv_path):
            cv_df = pd.read_csv(cv_path)
            y_col = 'F1-Score (Mean)' if 'F1-Score (Mean)' in cv_df.columns else cv_df.columns[1]
            fig_cv = px.bar(cv_df, x='Model', y=y_col, 
                            color='Model', color_discrete_sequence=px.colors.qualitative.Set2,
                            template='plotly_dark')
            fig_cv.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False, yaxis_range=[0.6, 1.0])
            st.plotly_chart(fig_cv, use_container_width=True)
        else:
            st.warning("Visualisasi cv_model_comparison.png tidak ditemukan.")
            
    st.write("---")
    
    col_eval3, col_eval4 = st.columns(2)
    
    with col_eval3:
        st.markdown("### Studi Ablasi Fitur (Feature Ablation Study)")
        st.write("Mengukur pengaruh penambahan satu fitur linguistik numerik secara individual ke dalam fitur baseline TF-IDF pada model SVM.")
        ab_path = 'outputs/results/feature_ablation_study.csv'
        if os.path.exists(ab_path):
            ab_df = pd.read_csv(ab_path)
            y_col_ab = 'Feature Configuration' if 'Feature Configuration' in ab_df.columns else ab_df.columns[0]
            # Ensure sort by F1-Score
            ab_df = ab_df.sort_values(by='F1-Score', ascending=True)
            fig_ab = px.bar(ab_df, x='F1-Score', y=y_col_ab, orientation='h',
                            color=y_col_ab, color_discrete_sequence=px.colors.qualitative.Set3,
                            template='plotly_dark')
            fig_ab.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False, xaxis_range=[0.85, 0.90])
            st.plotly_chart(fig_ab, use_container_width=True)
        else:
            st.warning("Visualisasi feature_ablation_study.png tidak ditemukan.")
            
    with col_eval4:
        st.markdown("### Tingkat Kepentingan Fitur (Feature Importance)")
        st.write("Mengukur signifikansi relatif 5 fitur linguistik numerik hasil rekayasa berdasarkan model ensemble Random Forest.")
        fi_path = 'outputs/results/feature_importance_rf.csv'
        if os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path)
            # Filter for engineered features or top 10 overall features for clarity
            eng_features = ['text_length', 'exclamation_count', 'uppercase_ratio', 'sentiment_score', 'lexical_diversity']
            fi_df_filtered = fi_df[fi_df['feature'].isin(eng_features)].sort_values(by='importance', ascending=True)
            if fi_df_filtered.empty:
                fi_df_filtered = fi_df.head(10).sort_values(by='importance', ascending=True)
                
            fig_fi = px.bar(fi_df_filtered, x='importance', y='feature', orientation='h',
                             color='feature', color_discrete_sequence=px.colors.qualitative.Pastel2,
                             template='plotly_dark')
            fig_fi.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Visualisasi feature_importance.png tidak ditemukan.")

# ==============================================================================
# TAB 4: METODOLOGI & DOKUMENTASI
# ==============================================================================
elif menu == "Metodologi & Dokumentasi":
    st.subheader("Metodologi & Alur Preprocessing")
    
    st.markdown("### Alur Kerja Inferensi Sistem (Pipeline)")
    st.write("Berikut diagram alur kerja pemrosesan data ulasan dari input pengguna mentah hingga keluar hasil prediksi keaslian ulasan.")
    if os.path.exists('outputs/figures/inference_flow_diagram.png'):
        st.image('outputs/figures/inference_flow_diagram.png', use_container_width=True)
    else:
        st.warning("Visualisasi inference_flow_diagram.png tidak ditemukan.")
        
    st.write("---")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        st.markdown("""
        ### Langkah Preprocessing Teks (Leksikal)
        *   **Case Folding**: Mengubah seluruh teks menjadi huruf kecil (*lowercasing*) agar kata dengan perbedaan kapitalisasi dibaca sama.
        *   **Cleaning Teks**: Menghapus angka dan tanda baca yang tidak bernilai semantik untuk klasifikasi menggunakan kamus string Python dan Regex.
        *   **Stopwords Removal**: Membuang kata-kata umum bahasa Inggris (seperti *the, is, in, at*) menggunakan korpus dari pustaka NLTK.
        *   **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan *Porter Stemmer* (contoh: *running/runs* menjadi *run*) untuk memperkecil ukuran dimensi kamus kosakata.
        """)
        
    with col_doc2:
        st.markdown("""
        ### Rekayasa 5 Fitur Linguistik (Feature Engineering)
        1.  **Panjang Ulasan (`text_length`)**: Jumlah total karakter teks ulasan asli.
        2.  **Tanda Seru (`exclamation_count`)**: Jumlah karakter tanda seru `!` yang merepresentasikan penekanan emosi.
        3.  **Rasio Kapital (`uppercase_ratio`)**: Rasio kata berhuruf besar penuh (*ALL-CAPS*) terhadap total kata dalam ulasan.
        4.  **Skor Sentimen VADER (`sentiment_score`)**: Skor polaritas emosi gabungan (*compound score*) di rentang -1 (sangat negatif) hingga +1 (sangat positif).
        5.  **Keragaman Kosakata (`lexical_diversity`)**: Rasio jumlah kosakata unik terhadap total kata (*Type-Token Ratio / TTR*).
        """)
        
    st.markdown("""
    ### Panduan Penggunaan Demo Model:
    1.  Buka tab **Demo Deteksi Ulasan**.
    2.  Ketik ulasan produk Anda pada kolom area teks (dalam Bahasa Inggris). Anda juga bisa memuat salah satu dari 4 contoh ulasan yang disediakan pada kotak pilihan di atasnya.
    3.  Tekan tombol **Mulai Analisis Keaslian**.
    4.  Sistem akan menampilkan kartu prediksi (berwarna hijau jika **ASLI / GENUINE** dan berwarna merah jika **PALSU / FAKE**).
    5.  Di bagian bawah kartu prediksi, Anda dapat memantau perbandingan karakteristik gaya penulisan ulasan yang dimasukkan terhadap rata-rata dataset, lengkap dengan peringatan deteksi pola kebahasaan bot/generator otomatis.
    """)
