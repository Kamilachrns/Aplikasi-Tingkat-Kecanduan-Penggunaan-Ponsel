import streamlit as st
import pandas as pd
import numpy as np
import joblib # Impor library pickle untuk memuat model

# --- Model Loading ---
# Dalam aplikasi nyata, Anda akan memuat model yang telah dilatih.
# Pastikan file 'regresi.pkl' berada di direktori yang sama dengan file Streamlit ini.
try:
    with open('regresi.pkl', 'rb') as f:
        model = joblib.load(f)
    st.sidebar.success("Model 'regresi.pkl' berhasil dimuat!")
except FileNotFoundError:
    st.sidebar.warning("File 'regresi.pkl' tidak ditemukan. Menggunakan model dummy untuk demonstrasi.")
    # Fallback ke model dummy jika 'regresi.pkl' tidak ditemukan
    class DummyModel:
        def predict(self, data_df):
            # Ini adalah model dummy; dalam kenyataan, ini akan melakukan prediksi yang sebenarnya
            # Kita akan mengembalikan nilai berdasarkan formula sederhana untuk demonstrasi
            age = data_df['Age'].iloc[0]
            daily_usage_hours = data_df['Daily_Usage_Hours'].iloc[0]
            phone_checks_per_day = data_df['Phone_Checks_Per_Day'].iloc[0]
            screen_time_before_bed = data_df['Screen_Time_Before_Bed'].iloc[0]
            parental_control = data_df['Parental_Control'].iloc[0] # 0 for No, 1 for Yes

            # Formula dummy untuk Tingkat Kecanduan Penggunaan Ponsel
            # Semakin tinggi jam penggunaan, cek ponsel, waktu layar sebelum tidur, semakin tinggi kecanduan.
            # Kontrol orang tua mengurangi kecanduan.
            predicted_addiction_level = (
                (daily_usage_hours * 2.5) +
                (phone_checks_per_day * 0.1) +
                (screen_time_before_bed * 1.5) -
                (parental_control * 5.0) + # Parental control reduces addiction
                (age * 0.05) # Age might have a small positive or negative effect
            )
            # Ensure the level is not negative
            predicted_addiction_level = max(0, predicted_addiction_level)
            return np.array([[predicted_addiction_level]])
    model = DummyModel()
except Exception as e:
    st.sidebar.error(f"Terjadi kesalahan saat memuat model: {e}. Menggunakan model dummy.")
    # Fallback ke model dummy jika ada kesalahan lain
    class DummyModel:
        def predict(self, data_df):
            age = data_df['Age'].iloc[0]
            daily_usage_hours = data_df['Daily_Usage_Hours'].iloc[0]
            phone_checks_per_day = data_df['Phone_Checks_Per_Day'].iloc[0]
            screen_time_before_bed = data_df['Screen_Time_Before_Bed'].iloc[0]
            parental_control = data_df['Parental_Control'].iloc[0]

            predicted_addiction_level = (
                (daily_usage_hours * 2.5) +
                (phone_checks_per_day * 0.1) +
                (screen_time_before_bed * 1.5) -
                (parental_control * 5.0) +
                (age * 0.05)
            )
            predicted_addiction_level = max(0, predicted_addiction_level)
            return np.array([[predicted_addiction_level]])
    model = DummyModel()
# --- End Model Loading ---


# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Prediksi Tingkat Kecanduan Ponsel",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📱 Aplikasi Prediksi Tingkat Kecanduan Penggunaan Ponsel")
st.markdown("""
Aplikasi ini memprediksi tingkat kecanduan penggunaan ponsel berdasarkan beberapa faktor.
Masukkan nilai di bawah ini untuk mendapatkan prediksi.
""")

st.write("---")

# --- Input Data Baru ---
st.header("Masukkan Data Baru")

try:
    new_age = st.number_input("Masukkan nilai Age (Usia):", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    new_daily_usage_hours = st.number_input("Masukkan nilai Daily Usage Hours (Jam Penggunaan Harian):", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    new_phone_checks_per_day = st.number_input("Masukkan nilai Phone Checks Per Day (Berapa kali memeriksa ponselnya dalam 1 hari):", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    new_screen_time_before_bed = st.number_input("Masukkan nilai Screen Time Before Bed (Waktu Menggunakan HP Sebelum Tidur - menit):", min_value=0.0, max_value=180.0, value=30.0, step=5.0)
    
    parental_control_option = st.radio(
        "Parental Control (Kontrol Orang Tua):",
        ("Tidak", "Ya"),
        index=0 # Default to "Tidak"
    )
    new_parental_control = 1.0 if parental_control_option == "Ya" else 0.0

    if st.button("Prediksi Tingkat Kecanduan"):
        # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
        new_data_df = pd.DataFrame(
            [[new_age, new_daily_usage_hours, new_phone_checks_per_day, new_screen_time_before_bed, new_parental_control]],
            columns=['Age', 'Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Screen_Time_Before_Bed', 'Parental_Control']
        )

        # Lakukan prediksi menggunakan model yang sudah dilatih
        predicted_addiction_level = model.predict(new_data_df)

        st.success(f"**Prediksi Tingkat Kecanduan Penggunaan Ponsel:**")
        st.info(f"Untuk Usia = {new_age} tahun, Jam Penggunaan Harian = {new_daily_usage_hours} jam, Cek Ponsel Per Hari = {new_phone_checks_per_day} kali, Waktu Layar Sebelum Tidur = {new_screen_time_before_bed} menit, dan Kontrol Orang Tua = {parental_control_option}:")
        # predicted_addiction_level adalah array 2D, ambil nilai tunggalnya
        st.success(f"Tingkat Kecanduan Penggunaan Ponsel diprediksi: **{predicted_addiction_level[0][0]:,.2f}**")

except ValueError:
    st.error("Input tidak valid. Harap masukkan angka.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

st.write("---")
st.markdown("Dikembangkan dengan ❤️ oleh Data Scientist Anda.")