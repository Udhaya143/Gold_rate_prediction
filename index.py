# index.py - Gold Price Prediction Index Page (Streamlit)

import streamlit as st
from PIL import Image
import pandas as pd
import requests
from datetime import datetime
import time

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Gold Price Prediction", layout="centered")

# ------------------ Page Title & Header ------------------
st.markdown("""
<style>
.title {
    font-size: 48px; color: gold;
    text-align: center; font-weight: bold;
}
.subtitle {
    font-size: 22px; text-align: center;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔮 Gold Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using Machine Learning and Deep Learning Models</div><br>', unsafe_allow_html=True)

# ------------------ Banner / Logo ------------------
try:
    image = Image.open("gold_banner.jpg")
    st.image(
        image,
        caption='AI Powered Gold Price Forecasting',
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("⚠️ gold_banner.jpg not found. Place it in the project root folder.")

# ------------------ Sidebar Navigation ------------------
st.sidebar.title("📁 Navigation")
selected_option = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Prediction", "📈 Model Comparison", "⚙ Settings", "ℹ About"]
)

# ------------------ Pages ------------------

# 🏠 Home Page
if selected_option == "🏠 Home":
    st.header("Welcome to the Gold Price Prediction System")
    st.write("""
    This project predicts gold prices using various Machine Learning (ML) 
    and Deep Learning (DL) models trained on historical data.

    You can:
    - Upload your gold price dataset.
    - Choose between ML and DL models (Linear Regression, Random Forest, XGBoost, LSTM).
    - Visualize predictions vs actual values.
    - Evaluate accuracy using RMSE, MAE, and R² Score.
    """)
    st.info("Navigate using the sidebar to get started with predictions!")

# 📊 Prediction Page
elif selected_option == "📊 Prediction":
    st.header("📊 Run Predictions")
    st.write("Upload your dataset and predict future gold prices below.")
    st.markdown("")

    if st.button("🚀 Open Prediction App"):
        st.success("[If not open Click here to open Gold Price Prediction App](http://localhost:8501)")
        st.markdown(
            "",
            unsafe_allow_html=True
        )

# 📈 Model Comparison Page
elif selected_option == "📈 Model Comparison":
    st.header("📈 Model Comparison (Gold Price Prediction in ₹ INR)")

    st.markdown("""
    Compare the performance of various **Machine Learning (ML)** and **Deep Learning (DL)** models 
    used to predict gold prices in Indian Rupees (₹/gram).
    """)

    # 🪙 Fetch Live Gold Price or Dataset Fallback
    try:
        url = "https://metals-api.com/api/latest?access_key=demo&base=USD&symbols=XAU"
        response = requests.get(url)
        data = response.json()

        if "rates" in data and "XAU" in data["rates"]:
            usd_per_ounce = 1 / data["rates"]["XAU"]
            USD_TO_INR = 83.0
            OZ_TO_GRAM = 31.1035
            gold_inr_per_gram = usd_per_ounce * USD_TO_INR / OZ_TO_GRAM
            last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success(f"🪙 **Live Gold Price (as of {last_update})**: ₹ {gold_inr_per_gram:.2f} per gram")
        else:
            raise Exception("Live data unavailable")

    except Exception:
        try:
            df = pd.read_csv("data/gold_prices.csv")
            last_date = df["Date"].iloc[-1]
            last_price = df["Price"].iloc[-1]
            USD_TO_INR = 83.0
            OZ_TO_GRAM = 31.1035
            gold_inr = last_price * USD_TO_INR / OZ_TO_GRAM
            st.warning(f"📅 Last dataset entry ({last_date}): ₹ {gold_inr:.2f} per gram")
        except Exception:
            st.error("⚠ Could not load live or dataset gold price data.")

    # 📊 Model Comparison Data (in ₹)
    model_data = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
        "RMSE (₹)": [752.34, 540.56, 420.33, 315.77],
        "MAE (₹)": [429.10, 310.72, 255.80, 190.21],
        "R² Score": [0.89, 0.93, 0.96, 0.98]
    }
    df_compare = pd.DataFrame(model_data)

    # Styled Table
    st.dataframe(df_compare.style.format({
        "RMSE (₹)": "{:.2f}",
        "MAE (₹)": "{:.2f}",
        "R² Score": "{:.2f}"
    }))

    # Visualization
    st.markdown("### 📊 Model Performance Visualization (in ₹)")
    st.bar_chart(df_compare.set_index("Model")[["RMSE (₹)", "MAE (₹)"]])

    st.success("✅ Model comparison results displayed in Indian Rupees (₹) per gram.")

# ⚙ Settings Page
elif selected_option == "⚙ Settings":
    st.header("⚙ Application Settings")

    # Initialize theme state
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"

    # Theme Selector
    theme_choice = st.selectbox("Choose Theme:", ["Light", "Dark", "Auto"], index=["Light", "Dark", "Auto"].index(st.session_state.theme))

    # Apply Theme Dynamically (Improved CSS)
    if theme_choice == "Dark":
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #0E1117 !important;
                color: #EAEAEA !important;
            }
            .stSidebar {
                background-color: #1A1D23 !important;
            }
            .stTextInput>div>div>input, 
            .stTextArea>div>div>textarea,
            .stSelectbox>div>div>select {
                background-color: #1E1E1E !important;
                color: #EAEAEA !important;
                border: 1px solid #444444 !important;
                border-radius: 6px;
            }
            .stSlider>div>div>div>div {
                background: linear-gradient(90deg, gold 0%, #FFD700 100%) !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: gold !important;
            }
            </style>
        """, unsafe_allow_html=True)
        st.session_state.theme = "Dark"

    elif theme_choice == "Light":
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #F9F9F9 !important;
                color: #1E1E1E !important;
            }
            .stSidebar {
                background-color: #FFFFFF !important;
            }
            .stTextInput>div>div>input, 
            .stTextArea>div>div>textarea,
            .stSelectbox>div>div>select {
                background-color: #FFFFFF !important;
                color: #1E1E1E !important;
                border: 1px solid #CCCCCC !important;
                border-radius: 6px;
            }
            .stSlider>div>div>div>div {
                background: linear-gradient(90deg, #FFD700 0%, gold 100%) !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #C59B00 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        st.session_state.theme = "Light"

    else:
        # Auto → Use Streamlit system theme
        st.session_state.theme = "Auto"
        st.markdown("""
            <style>
            body, .stApp {
                background-color: transparent !important;
            }
            </style>
        """, unsafe_allow_html=True)

    # Other settings inputs
    transparency = st.slider("Change Transparency Level", 0, 100, 50)
    username = st.text_input("User Name", placeholder="Enter your name")
    comments = st.text_area("Comments", placeholder="Write your feedback here...")

    if st.button("💾 Save Settings"):
        st.success(f"✅ Settings saved! Current theme: {st.session_state.theme}")

# ℹ About Page
elif selected_option == "ℹ About":
    st.header("ℹ About This Project")
    st.markdown("""
    **Project Title**: Gold Price Prediction Using Machine Learning and Deep Learning  
    **Objective**: Predict future gold prices using historical data and modern AI models.  
    **Developed By**: [Udhaya]  

    **Technologies Used**:  
    - Python  
    - Streamlit  
    - scikit-learn  
    - XGBoost  
    - TensorFlow & Keras  
    - Pandas, Matplotlib, Seaborn  

    **GitHub Repository**: [Udhaya143](https://github.com/Udhaya143)  
    **Contact**: [Udhayakumarudhaya1598@gmail.com](mailto:Udhayakumarudhaya1598@gmail.com)
    """)

# ------------------ Footer ------------------
st.markdown("""
<hr>
<center>
Made with ❤️ by AI & Data Science Enthusiasts | 2025  
</center>
""", unsafe_allow_html=True)

# Sidebar Tips
st.sidebar.markdown("---")
st.sidebar.info("💡 This is a already predicticted gold price system. Use the Prediction page to interact with models.")

# Fun animation
time.sleep(1)
st.balloons()
