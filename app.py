import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# Hanya memuat model sekali saat aplikasi dimulai
if 'model' not in st.session_state:
    st.session_state.model = joblib.load('model_mlp.pkl')  # Memuat model dengan nama model_mlp.pkl
    st.session_state.scaler = joblib.load('scaler.pkl')  # Memuat scaler dengan nama scaler.pkl

# Menggunakan model dan scaler yang sudah dimuat
model = st.session_state.model
scaler = st.session_state.scaler

# Pengaturan halaman
st.set_page_config(page_title="Diabetes", layout="wide")

# Judul dan Deskripsi
st.title("GlucoSmart")
st.write("""
This web predicts the risk of diabetes based on user input.
""")

# Pilih model yang digunakan
add_selectitem = st.sidebar.selectbox("Pilih model yang digunakan", ("Prediksi Diabetes ANN"))

def diabetes():
    st.write("""
    Situs ini memprediksi risiko diabetes!
    """)
    
    # Input data manual
    st.sidebar.header("Input data Anda secara manual")
    
    # Mengganti slider dengan input teks
    Pregnancies = st.sidebar.number_input("Berapa Kali Hamil", min_value=0, max_value=50, value=25)
    Glucose = st.sidebar.number_input("Tingkat Glukosa", min_value=0, max_value=500, value=100)
    BloodPressure = st.sidebar.number_input("Tekanan Darah", min_value=0, max_value=500, value=90)
    SkinThickness = st.sidebar.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
    Insulin = st.sidebar.number_input("Tingkat Insulin", min_value=0, max_value=500, value=100)
    BMI = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    
    # Mengubah input 'Apakah ada keturunan diabetes?' menjadi pilihan "Ada" atau "Tidak"
    diabetes_ancestry = st.sidebar.radio("Apakah ada keturunan diabetes?", ("Tidak", "Ada"))
    DiabetesPedigreeFunction = 1 if diabetes_ancestry == "Ada" else 0  # Memberikan nilai 1 jika "Ada" dan 0 jika "Tidak"
    
    Age = st.sidebar.number_input("Usia", min_value=18, max_value=120, value=30)

    # Membuat dataframe dari input yang diterima
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,  # Menggunakan nilai 0 atau 1
            'Age': Age}

    df = pd.DataFrame(data, index=[0])

    # Proses normalisasi data dengan scaler yang sudah dimuat
    df_scaled = scaler.transform(df)

    # Prediksi menggunakan model yang sudah dimuat
    if st.sidebar.button("Prediksi"):
        st.write("Data yang digunakan untuk prediksi:")
        st.write(df)

        # Prediksi menggunakan model yang sudah dilatih
        pred = model.predict(df_scaled)
        
        
        # Menampilkan hasil prediksi
        st.subheader("Prediksi")
        with st.spinner("Memprediksi..."):
            time.sleep(2)
        if pred[0] == 0:
            st.error("Diabetes sitik ngkas modar")
        else:
            st.success("ANDA BEBAS DARI DIABETES")
# Menjalankan fungsi untuk prediksi diabetes
if add_selectitem == "Prediksi Diabetes ANN":
    diabetes()
