import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN & THEME (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Student Academic Risk EWS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. OPTIMASI MEMORI (Caching Objek ML)
# ==========================================
@st.cache_resource
def load_ml_components():
    model_dir = os.path.join(os.path.dirname(__file__), '../model')
    
    with open(os.path.join(model_dir, 'kprototypes_model.pkl'), 'rb') as f:
        kproto = pickle.load(f)
    with open(os.path.join(model_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_dir, 'classifier_model.pkl'), 'rb') as f:
        classifier = pickle.load(f)
        
    return kproto, encoder, classifier

try:
    kproto_model, data_encoder, classifier_model = load_ml_components()
    shap_explainer = shap.TreeExplainer(classifier_model)
except Exception as e:
    st.error(f"Gagal memuat komponen model penunjang. Pastikan file .pkl tersedia di folder 'model/'. Error: {e}")

# ==========================================
# 3. NAVIGASI UTAMA (Sidebar Adaptive)
# ==========================================
with st.sidebar:
    st.title("🎓 EWS Dashboard")
    selected_page = option_menu(
        menu_title="Menu Utama",
        options=["Home", "Panduan", "Dataset Overview", "Risk Prediction", "Visualization", "About"],
        icons=["house", "book", "table", "shield-exclamation", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px!", "background-color": "transparent"}, 
            "icon": {"color": "#3b82f6", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "var(--text-color)"},
            "nav-link-selected": {"background-color": "#1e3a8a", "color": "white"},
        }
    )

# ==========================================
# 4. ROUTING HALAMAN
# ==========================================

# --- HALAMAN 1: HOME ---
if selected_page == "Home":
    st.title("🏛️ Early Warning System: Deteksi Risiko Akademik Siswa")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Deskripsi Proyek")
        st.write(
            "Aplikasi berbasis analitik prediktif ini dirancang khusus untuk membantu pihak sekolah, "
            "khususnya guru Bimbingan Konseling (BK), dalam memetakan profil perilaku hidup siswa dan "
            "mendeteksi potensi kegagalan akademik pada mata pelajaran Bahasa Portugis secara dini."
        )
        st.write(
            "Sistem ini bekerja dengan memadukan dua pendekatan Data Mining secara berkesinambungan. "
            "Model pertama melakukan segmentasi gaya hidup siswa menggunakan algoritma **K-Prototypes**, "
            "kemudian klaster tersebut dijadikan salah satu indikator bagi algoritma **XGBoost** untuk "
            "memprediksi probabilitas risiko kelulusan tanpa memanfaatkan riwayat nilai ujian sebelumnya."
        )
    with col2:
        st.info(
            "**Identitas Kelompok:**\n"
            "- 25051905013 ABDUL KHAHAR\n"
            "- 25051905012 SAMUEL BRAEN LATUMAHINA\n\n"
            "**Mata Kuliah:**\n"
            "Data Mining - S2 Informatika\n"
            "Universitas Negeri Surabaya"
        )

# --- HALAMAN 2: PANDUAN PENGGUNAAN ---
elif selected_page == "Panduan":
    st.title("📖 Panduan Operasional Sistem")
    st.markdown("---")
    
    st.markdown("""
    Sistem ini dirancang sebagai alat pendukung keputusan (*decision support system*) terintegrasi. Dasbor ini tidak menggantikan peran konselor, melainkan menyoroti variabel gaya hidup tersembunyi yang berisiko menggagalkan kelulusan akademik siswa.

    ### 1. Fungsi Navigasi Menu Utama
    * **Home:** Beranda utama yang memuat identitas pengembang dan latar belakang proyek.
    * **Panduan:** Halaman instruksi operasional yang sedang Anda baca.
    * **Dataset Overview:** Halaman transparansi data. Menampilkan metrik statistik, proporsi kelas target, dan sampel data mentah yang digunakan untuk melatih kecerdasan buatan.
    * **Risk Prediction:** Mesin utama aplikasi. Tempat memasukkan data spesifik seorang siswa untuk dievaluasi tingkat risiko kegagalannya secara *real-time*.
    * **Visualization:** Ruang eksplorasi interaktif. Memuat grafik analitik untuk melihat korelasi antar fitur sosial dan dampaknya terhadap performa akademik secara umum.
    * **About:** Dokumentasi teknis metodologi. Memuat penjelasan algoritma (K-Prototypes & XGBoost), arsitektur CRISP-DM, dan metrik evaluasi model.

    ### 2. Alur Pengujian Prediksi Risiko
    Gunakan halaman **Risk Prediction** untuk mengevaluasi siswa dengan mengikuti prosedur operasional berikut:
    1. **Pengumpulan Data:** Siapkan data demografi dan catatan perilaku siswa (absensi, jam belajar, riwayat kegagalan sebelumnya).
    2. **Input Parameter:** Masukkan data siswa ke dalam form interaktif yang tersedia. Pastikan variabel numerik seperti jumlah absensi diisi dengan angka aktual semester berjalan.
    3. **Eksekusi Sistem:** Tekan tombol **Analisis Risiko Siswa**. Sistem akan memproses data melalui model K-Prototypes untuk penentuan profil gaya hidup, dilanjutkan ke model XGBoost untuk kalkulasi probabilitas gagal.
    4. **Evaluasi Indikator Utama:** Periksa panel **Probabilitas Risiko Gagal**. Jika probabilitas melampaui ambang batas (*threshold* 0.30), sistem akan membunyikan alarm peringatan dini (merah). Jika di bawah ambang batas, siswa dinyatakan aman (hijau).
    5. **Interpretasi Explainable AI (SHAP):** Jangan berhenti pada angka probabilitas. Analisis grafik *Waterfall* di bagian bawah untuk menyusun strategi intervensi yang tepat sasaran.

    ### 3. Panduan Pembacaan Visualisasi & Indikator
    * **Tombol Analisis:** Memerintahkan server untuk mengeksekusi *pipeline* komputasi mesin pembelajaran ganda.
    * **Indikator Merah (Peringatan):** Siswa memiliki probabilitas gagal tinggi berdasarkan pola perilakunya saat ini. Diperlukan tindakan preventif segera.
    * **Indikator Hijau (Aman):** Siswa memiliki fondasi gaya hidup dan dukungan sosial yang cukup untuk mendukung kelulusan akademiknya.
    * **Grafik SHAP - Bar Merah:** Variabel yang mendorong siswa ke arah kegagalan. Angka di dalam bar menunjukkan besaran kontribusi negatifnya. Ini adalah titik fokus utama untuk konseling (misalnya: tingginya absensi atau rendahnya motivasi pendidikan lanjutan).
    * **Grafik SHAP - Bar Biru:** Variabel pelindung yang menahan risiko gagal siswa. Fitur positif ini (misalnya: keikutsertaan kegiatan ekstrakurikuler) harus dipertahankan.
    """)

# --- HALAMAN 3: DATASET OVERVIEW ---
elif selected_page == "Dataset Overview":
    st.title("📊 Ringkasan Eksplorasi Data (Dataset Overview)")
    st.markdown("---")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sampel Data", "649 Baris", "Kategori: Tabular")
    m2.metric("Total Fitur Evaluasi", "30 Atribut Sosial", "Eksklusi: G1 & G2")
    m3.metric("Proporsi Kelas Berisiko", "19.26%", "Kondisi: Imbalanced")

    st.subheader("Sampel Data Terproses")
    try:
        df_display = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed/X_train_clustered.csv'))
        st.dataframe(df_display.head(10), use_container_width=True)
    except:
        st.warning("Gunakan notebook untuk mengekspor data terproses ke folder processed terlebih dahulu.")

# --- HALAMAN 4: RISK PREDICTION ---
elif selected_page == "Risk Prediction":
    st.title("🎯 Form Evaluasi Risiko Akademik Mandiri")
    st.markdown("---")
    
    st.subheader("Input Parameter Karakteristik Siswa")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            school = st.selectbox("Asal Sekolah", ["GP", "MS"])
            sex = st.selectbox("Jenis Kelamin", ["F", "M"])
            age = st.slider("Usia Siswa", 15, 22, 17)
            studytime = st.slider("Waktu Belajar Mingguan (Kategori 1-4)", 1, 4, 2)
            failures = st.slider("Jumlah Riwayat Gagal Kelas Sebelumnya", 0, 3, 0)
            
        with c2:
            higher = st.selectbox("Ingin Melanjutkan Kuliah?", ["yes", "no"])
            absences = st.number_input("Jumlah Absensi / Bolos Semester Ini", min_value=0, max_value=100, value=0)
            Dalc = st.slider("Konsumsi Alkohol Hari Kerja (Skala 1-5)", 1, 5, 1)
            Walc = st.slider("Konsumsi Alkohol Akhir Pekan (Skala 1-5)", 1, 5, 1)
            goout = st.slider("Intensitas Bermain Bersama Teman (Skala 1-5)", 1, 5, 3)

        with c3:
            address = st.selectbox("Tipe Alamat Tinggal", ["U", "R"])
            famsize = st.selectbox("Ukuran Keluarga", ["LE3", "GT3"])
            Pstatus = st.selectbox("Status Tinggal Orang Tua", ["T", "A"])
            Medu = st.slider("Tingkat Pendidikan Ibu (Skala 0-4)", 0, 4, 2)
            Fedu = st.slider("Tingkat Pendidikan Ayah (Skala 0-4)", 0, 4, 2)
            
        submit_btn = st.form_submit_button("Analisis Risiko Siswa")
        
    if submit_btn:
        st.markdown("---")
        st.subheader("Hasil Analisis Prediktif")
        
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
            'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': 'other', 'Fjob': 'other',
            'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'studytime': studytime,
            'failures': failures, 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no', 'activities': 'yes',
            'nursery': 'yes', 'higher': higher, 'internet': 'yes', 'romantic': 'no', 'famrel': 4,
            'freetime': 3, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc, 'health': 3, 'absences': absences
        }
        df_input = pd.DataFrame([input_data])
        
        try:
            # 1. Jalankan Tahap 1: Prediksi Klaster Gaya Hidup
            cluster_assigned = kproto_model.predict(df_input.values, categorical=[0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22])
            
            # PERBAIKAN 1: Kembalikan lifestyle_cluster menjadi Integer agar dikenali Encoder
            df_input['lifestyle_cluster'] = int(cluster_assigned[0])
            
            # Pisahkan 17 kolom kategorikal asli
            original_cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                                 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                                 'nursery', 'higher', 'internet', 'romantic']
            
            # Paksa 17 kolom asli ini menjadi string (sesuai format saat dilatih K-Prototypes)
            df_input[original_cat_cols] = df_input[original_cat_cols].astype(str)
            
            # Gabungkan dengan lifestyle_cluster untuk OneHotEncoder
            expected_cat_cols = original_cat_cols + ['lifestyle_cluster']
            expected_num_cols = [col for col in df_input.columns if col not in expected_cat_cols]
            
            # 2. Jalankan Tahap 2: Preprocessing One-Hot Encoding
            encoded_cats = data_encoder.transform(df_input[expected_cat_cols])
            encoded_cats_df = pd.DataFrame(encoded_cats, columns=data_encoder.get_feature_names_out(expected_cat_cols))
            
            df_final_encoded = encoded_cats_df.copy()
            df_final_encoded[expected_num_cols] = df_input[expected_num_cols]
            
            # PERBAIKAN 2: Tarik nama fitur asli dari objek model XGBoost, bukan dari SHAP
            if hasattr(classifier_model, 'feature_names_in_'):
                expected_features = classifier_model.feature_names_in_
            else:
                expected_features = classifier_model.get_booster().feature_names
                
            df_final_encoded = df_final_encoded[expected_features]
            
            # 3. Jalankan Tahap 3: Prediksi Probabilitas Klasifikasi
            risk_probability = classifier_model.predict_proba(df_final_encoded)[:, 1][0]
            
            # Penerapan Custom Threshold Teroptimasi (0.3)
            CRITICAL_THRESHOLD = 0.30
            is_risk = risk_probability >= CRITICAL_THRESHOLD
            
            # Menampilkan Output Secara Visual
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Probabilitas Risiko Gagal", f"{risk_probability*100:.2f}%", f"Threshold Konfigurasi: {CRITICAL_THRESHOLD}")
                if is_risk:
                    st.error("🚨 KEPUTUSAN: SISTEM MENYALAKAN ALARM PERINGATAN DINI (RISIKO TINGGI)")
                else:
                    st.success("✅ KEPUTUSAN: SISWA DINYATAKAN AMAN / PROBABILITAS KELULUSAN TINGGI")
                    
            with col_res2:
                # Tentukan deskripsi berdasarkan hasil analisis EDA Anda sebelumnya
                # Pemetaan semantik deskripsi klaster dengan isyarat visual (UX)
                cluster_descriptions = {
                    0: "✅ AMAN: Disiplin tinggi, tingkat absensi sangat rendah (mendekati nol), dan konsumsi alkohol terkendali. Profil siswa dengan fondasi akademik yang aman.",
                    1: "⚖️ MODERAT: Tingkat absensi menengah, gaya hidup sosial standar, serta waktu belajar dan performa akademik yang cukup stabil.",
                    2: "🚨 RISIKO TINGGI: Ditandai dengan tingkat absensi ekstrem (>12 kali), tingginya konsumsi alkohol di akhir pekan, dan indikasi masalah kedisiplinan kronis."
                }

                klaster_id = int(cluster_assigned[0])
                st.info(f"**Profil Gaya Hidup (Klaster {klaster_id}):** {cluster_descriptions.get(klaster_id, 'Tidak terdefinisi')}")
                st.caption("Karakteristik klaster dapat diidentifikasi pada pola segmentasi.")
                
            # 4. MENAMPILKAN EXPLAINABLE AI (SHAP)
            st.markdown("---")
            st.subheader("💡 Transparansi Keputusan Model (Explainable AI - SHAP)")
            st.write("Grafik di bawah menjelaskan fitur apa saja yang berkontribusi:")
            
            shap_values = shap_explainer(df_final_encoded)
            fig, ax = plt.subplots(figsize=(10, 5))
            
            fig.patch.set_facecolor('white') 
            ax.set_facecolor('white')
            
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

            # ==========================================
            # 5. INTERPRETASI DAN NARASI OTOMATIS (DYNAMIC INTERPRETATION)
            # ==========================================
            st.markdown("---")
            st.subheader("📝 Kesimpulan & Rekomendasi Intervensi Konseling")
            
            # Membuat penampung untuk poin-poin analisis
            faktor_risiko = []
            faktor_pelindung = []
            
            # Ekstraksi otomatis berbasis aturan terhadap input kritis siswa
            if absences > 10:
                faktor_risiko.append(f"**Tingkat Absensi Tinggi ({absences} kali bolos):** Perilaku membolos ini menjadi pemicu utama renggangnya keterikatan akademis siswa di kelas.")
            if higher == "no":
                faktor_risiko.append("**Ketiadaan Motivasi Studi Lanjut (higher = no):** Tidak adanya target masa depan menurunkan daya juang belajar siswa secara drastis.")
            if failures > 0:
                faktor_risiko.append(f"**Rekam Jejak Kegagalan Akademis ({failures} kali tinggal kelas):** Ada pola kerentanan akademis masa lalu yang belum terselesaikan dengan baik.")
            if goout >= 4:
                faktor_risiko.append(f"**Intensitas Bermain Terlalu Tinggi (Skala {goout}/5):** Alokasi waktu luang di luar rumah bersama teman mengorbankan waktu evaluasi mandiri.")
            if Dalc >= 3 or Walc >= 3:
                faktor_risiko.append(f"**Pola Konsumsi Alkohol Akhir Pekan/Hari Kerja (Skala Walc: {Walc}/5):** Gaya hidup bebas berisiko menurunkan kebugaran fisik dan konsentrasi belajar.")
                
            if studytime >= 3:
                faktor_pelindung.append(f"**Komitmen Waktu Belajar (Skala {studytime}/4):** Alokasi waktu belajar mandiri yang baik menjadi modal utama siswa untuk bertahan.")
            if failures == 0:
                faktor_pelindung.append("**Ketiadaan Riwayat Gagal Kelas:** Siswa tidak memiliki beban psikologis atau ketertinggalan materi dari jenjang sebelumnya.")
                
            # Menampilkan narasi penjelas berdasarkan keputusan model
            if is_risk:
                st.write(
                    f"Berdasarkan visualisasi kontribusi fitur di atas, model menetapkan status **Risiko Tinggi** "
                    f"karena akumulasi faktor negatif (bar berwarna merah) mendominasi pertimbangan algoritma. "
                    f"Berikut adalah rincian alasan utama di balik keputusan model untuk siswa ini:"
                )
                
                for item in faktor_risiko:
                    st.markdown(f"- {item}")
                    
                st.markdown("### 📋 Rekomendasi Tindakan Guru BK:")
                if higher == "no":
                    st.warning("👉 **Intervensi Motivasi:** Lakukan konseling personal untuk menggali alasan di balik keengganan melanjutkan studi, dan bantu siswa memetakan target karier jangka pendek.")
                if absences > 10:
                    st.warning("👉 **Kontrol Kedisiplinan:** Segera lakukan pemanggilan orang tua untuk mengklarifikasi angka absensi dan menyusun pakta integritas kehadiran.")
                if len(faktor_risiko) == 0:
                    st.warning("👉 **Evaluasi Komprehensif:** Faktor pemicu bersifat akumulatif dari variabel sosial kecil lainnya. Periksa detail grafik SHAP untuk melihat kombinasi fitur pendukung.")
            else:
                st.write(
                    f"Model menetapkan status **Aman** karena faktor pelindung internal dan sosial siswa "
                    f"(bar berwarna biru) berhasil meredam potensi risiko kegagalan. Alasan utama stabilitas akademik siswa ini meliputi:"
                )
                
                if len(faktor_pelindung) > 0:
                    for item in faktor_pelindung:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- **Kombinasi Gaya Hidup Seimbang:** Variabel sosial, pola asuh, dan pemanfaatan waktu luang siswa berada pada koridor aman berdasarkan komparasi historis dataset.")
                
                st.markdown("### 📋 Rekomendasi Tindakan Guru BK:")
                st.success("👉 Tetap lakukan pemantauan berkala dan pertahankan motivasi serta iklim suportif di lingkungan belajar siswa.")
            
        except Exception as e:
            import traceback
            st.error("Terjadi kesalahan sistematis pada arsitektur pipeline:")
            # PERBAIKAN 3: Jika masih ada error, tampilkan traceback secara utuh agar ketahuan detailnya
            st.code(traceback.format_exc())

# --- HALAMAN 5: VISUALIZATION ---
elif selected_page == "Visualization":
    st.title("📈 Dashboard Analitik Sosial & Profil Klaster")
    st.markdown("---")
    
    st.write(
        "Halaman ini mengeksplorasi hubungan antara faktor gaya hidup dan hasil akhir akademik, "
        "serta membedah karakteristik klaster yang dihasilkan oleh model *Unsupervised Learning* (K-Prototypes)."
    )
    
    try:
        # ==========================================
        # BAGIAN A: ANALISIS KORELASI GAYA HIDUP & KELULUSAN
        # ==========================================
        st.subheader("A. Dampak Fitur Sosial Terhadap Status Kelulusan")
        df_viz = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/raw/student-por.csv'), sep=';')
        df_viz['Status_Kelulusan'] = np.where(df_viz['G3'] >= 10, 'Aman (Lulus)', 'Berisiko (Gagal)')
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            fig1 = px.scatter(
                df_viz, x="absences", y="G3", color="Status_Kelulusan",
                color_discrete_map={"Aman (Lulus)": "#2563eb", "Berisiko (Gagal)": "#dc2626"},
                title="1. Dampak Absensi terhadap Nilai Akhir (G3)",
                labels={"absences": "Jumlah Absensi", "G3": "Nilai Akhir (0-20)"},
                opacity=0.7
            )
            fig1.add_hline(y=10, line_dash="dash", line_color="red")
            st.plotly_chart(fig1, use_container_width=True)
            
            risk_by_alc = df_viz.groupby('Dalc')['Status_Kelulusan'].value_counts(normalize=True).unstack().fillna(0)
            if 'Berisiko (Gagal)' in risk_by_alc.columns:
                risk_by_alc['% Berisiko'] = risk_by_alc['Berisiko (Gagal)'] * 100
                fig3 = px.bar(
                    risk_by_alc, y='% Berisiko', text_auto='.2f',
                    title="3. Konsumsi Alkohol vs Persentase Risiko",
                    labels={"Dalc": "Konsumsi Alkohol Harian (1-5)", "value": "Persentase Risiko (%)"},
                    color_discrete_sequence=["#ef4444"]
                )
                st.plotly_chart(fig3, use_container_width=True)

        with col_viz2:
            fig2 = px.box(
                df_viz, x="studytime", y="G3", color="studytime",
                title="2. Waktu Belajar vs Performa Akademik",
                labels={"studytime": "Waktu Belajar (1=Terendah, 4=Tertinggi)", "G3": "Nilai Akhir"},
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig2.add_hline(y=10, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
            
            fig4 = px.histogram(
                df_viz, x="failures", color="Status_Kelulusan", barmode="group",
                title="4. Pengaruh Riwayat Tinggal Kelas",
                color_discrete_map={"Aman (Lulus)": "#3b82f6", "Berisiko (Gagal)": "#ef4444"},
                labels={"failures": "Jumlah Gagal Masa Lalu", "count": "Jumlah Siswa"}
            )
            st.plotly_chart(fig4, use_container_width=True)
            
        st.markdown("---")
        
        # ==========================================
        # BAGIAN B: VISUALISASI PROFIL KLASTER (K-PROTOTYPES)
        # ==========================================
        st.subheader("B. Analisis Profil Klaster Gaya Hidup (K-Prototypes)")
        st.write("Visualisasi di bawah membedah karakteristik utama dari 3 klaster yang secara otomatis dibentuk oleh algoritma berdasarkan kemiripan pola hidup siswa.")
        
        # Memuat data yang sudah diklasterisasi
        df_clustered = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed/X_train_clustered.csv'))
        
        # Mapping nama klaster agar informatif di grafik (Sesuai analisis EDA)
        cluster_map = {0: "0 (Aman)", 1: "1 (Moderat)", 2: "2 (Risiko Tinggi)"}
        df_clustered['Nama_Klaster'] = df_clustered['lifestyle_cluster'].map(cluster_map)
        
        # Konversi failures ke string agar histogram memisahkan warnanya secara diskrit
        df_clustered['failures_str'] = df_clustered['failures'].astype(str)
        
        col_c1, col_c2 = st.columns(2)
        
        # Palet warna khusus untuk klaster agar konsisten (Aman: Hijau, Moderat: Oranye, Risiko: Merah)
        cluster_colors = {"0 (Aman)": "#10b981", "1 (Moderat)": "#f59e0b", "2 (Risiko Tinggi)": "#ef4444"}

        with col_c1:
            # Grafik Boxplot Absensi
            fig_c1 = px.box(
                df_clustered, x="Nama_Klaster", y="absences", color="Nama_Klaster",
                color_discrete_map=cluster_colors,
                title="1. Distribusi Absensi per Klaster",
                labels={"Nama_Klaster": "Kategori Klaster", "absences": "Jumlah Absensi"}
            )
            st.plotly_chart(fig_c1, use_container_width=True)
            
            # Grafik Boxplot Waktu Belajar
            fig_c3 = px.box(
                df_clustered, x="Nama_Klaster", y="studytime", color="Nama_Klaster",
                color_discrete_map=cluster_colors,
                title="3. Waktu Belajar Mingguan (Skala 1-4)",
                labels={"Nama_Klaster": "Kategori Klaster", "studytime": "Kategori Waktu Belajar"}
            )
            st.plotly_chart(fig_c3, use_container_width=True)

        with col_c2:
            # Grafik Boxplot Alkohol Akhir Pekan
            fig_c2 = px.box(
                df_clustered, x="Nama_Klaster", y="Walc", color="Nama_Klaster",
                color_discrete_map=cluster_colors,
                title="2. Konsumsi Alkohol Akhir Pekan (Skala 1-5)",
                labels={"Nama_Klaster": "Kategori Klaster", "Walc": "Tingkat Konsumsi (Walc)"}
            )
            st.plotly_chart(fig_c2, use_container_width=True)
            
            # Grafik Histogram Riwayat Gagal
            fig_c4 = px.histogram(
                df_clustered, x="Nama_Klaster", color="failures_str", barmode="group",
                title="4. Riwayat Tinggal Kelas per Klaster",
                labels={"Nama_Klaster": "Kategori Klaster", "failures_str": "Jumlah Kegagalan"},
                color_discrete_sequence=px.colors.sequential.Reds
            )
            st.plotly_chart(fig_c4, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memuat visualisasi. Pastikan dataset raw dan processed tersedia. Error: {e}")

# --- HALAMAN 6: ABOUT ---
elif selected_page == "About":
    st.title("ℹ️ Dokumentasi & Metodologi Arsitektur")
    st.markdown("---")
    
    st.markdown("""
    ### 1. Kerangka Kerja: CRISP-DM
    Proyek ini dibangun mematuhi standar *Cross-Industry Standard Process for Data Mining* (CRISP-DM), yang memastikan solusi berakar pada penyelesaian masalah dunia nyata (Business Understanding) sebelum memasuki fase rekayasa komputasi.
    * **Fokus Business Understanding:** Menggeser paradigma dari "memprediksi angka ujian" menjadi "membangun sistem peringatan dini (EWS)" yang memberikan ruang waktu bagi intervensi konseling.
    * **Mitigasi Data Leakage:** Secara sengaja menghapus fitur **G1** dan **G2** (nilai tengah semester) pada tahap *Data Preparation*. Pendekatan ini mencegah model menebak hasil secara matematis dan memaksa model mengekstraksi pola dari variabel sosiologis.

    ### 2. Algoritma 1: K-Prototypes Clustering
    Data demografi manusia pada umumnya terdiri dari variabel campuran (kategorikal dan numerik). 
    * Penggunaan *K-Means* tradisional sangat dihindari karena jarak *Euclidean* mendistorsi makna dari variabel kategorikal (seperti status pekerjaan orang tua).
    * Algoritma **K-Prototypes** diimplementasikan karena secara matematis menggabungkan perhitungan jarak *Euclidean* untuk atribut numerik dan *Hamming Distance* untuk pencocokan atribut kategorikal. 
    * Melalui *Elbow Method Cost*, dataset ini disegmentasi menjadi $K=3$ profil gaya hidup utama yang optimal.

    ### 3. Algoritma 2: Extreme Gradient Boosting (XGBoost)
    Prediktor utama dibangun menggunakan paradigma *Tree-Ensemble Learning*.
    * **Cost-Sensitive Learning:** Mengingat dataset bersifat imbalanced (minoritas siswa gagal), model XGBoost dirancang lebih peka terhadap kelas minoritas menggunakan kompensasi bobot (*scale_pos_weight*) yang dikombinasikan dengan teknik *Synthetic Minority Over-sampling Technique* (SMOTE).
    * **Threshold Moving:** Ambang batas probabilitas (*decision threshold*) diturunkan menjadi **$0.30$**. Dalam ranah mitigasi risiko, nilai *Recall* lebih diutamakan dibandingkan *Accuracy*. Mengorbankan akurasi untuk mendeteksi lebih banyak siswa berisiko (meminimalkan *False Negative*) adalah keputusan arsitektural yang rasional.

    ### 4. Transparansi Model: SHAP (Explainable AI)
    Kepercayaan pengguna (*user trust*) adalah metrik keberhasilan implementasi sistem. Model *black-box* seperti XGBoost didekonstruksi menggunakan algoritma **SHapley Additive exPlanations (SHAP)** yang mengakar pada teori permainan (*Game Theory*). SHAP memecah keputusan prediksi menjadi bobot kontribusi individual dari tiap fitur, memungkinkan Guru BK untuk melihat dengan akurat variabel gaya hidup mana yang menarik seorang siswa ke arah kegagalan.
    """)