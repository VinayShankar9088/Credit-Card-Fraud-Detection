import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.title("💳 Fraud Detection")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Bulk Prediction",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model : Logistic Regression")

st.sidebar.info(
"""
Dataset

✔ Kaggle Credit Card Fraud

Features

✔ Time
✔ V1 - V28
✔ Amount
"""
)

# -------------------- HOME PAGE --------------------
if menu == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection Dashboard")

    st.write(
        """
        Welcome to the Credit Card Fraud Detection Dashboard.

        This application uses a trained Logistic Regression model
        to detect fraudulent credit card transactions.

        ### Features

        ✅ Bulk CSV Prediction

        ✅ Fraud Detection

        ✅ Fraud Statistics

        ✅ Interactive Charts

        ✅ Download Prediction Report
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Logistic Regression")
    col2.metric("Framework", "Streamlit")
    col3.metric("Dataset", "Kaggle")

    st.markdown("---")

    st.info(
        "Use the **Bulk Prediction** page from the sidebar to upload a CSV file and detect fraudulent transactions."
    )
    
    # -------------------- BULK PREDICTION --------------------
elif menu == "📂 Bulk Prediction":

    st.title("📂 Bulk Transaction Fraud Detection")

    uploaded_file = st.file_uploader(
        "Upload Credit Card CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Dataset")

        st.dataframe(df.head())

        if st.button("🚀 Predict Fraud Transactions"):

            try:

                # Remove Class column if present
                if "Class" in df.columns:
                    X = df.drop("Class", axis=1)
                else:
                    X = df.copy()

                # Scale Data
                X_scaled = scaler.transform(X)

                # Prediction
                prediction = model.predict(X_scaled)

                probability = model.predict_proba(X_scaled)[:,1]

                # Add Results
                df["Prediction"] = prediction
                df["Fraud Probability"] = np.round(probability*100,2)

                df["Prediction"] = df["Prediction"].replace({
                    0:"✅ Genuine",
                    1:"🚨 Fraud"
                })

                fraud = (df["Prediction"]=="🚨 Fraud").sum()
                genuine = (df["Prediction"]=="✅ Genuine").sum()
                total = len(df)

                fraud_percent = round((fraud/total)*100,2)

                st.success("Prediction Completed Successfully ✅")

                # ---------------- KPI ----------------

                c1,c2,c3,c4 = st.columns(4)

                c1.metric("Transactions",total)
                c2.metric("Fraud",fraud)
                c3.metric("Genuine",genuine)
                c4.metric("Fraud %",f"{fraud_percent}%")

                st.markdown("---")

                # Show Fraud Only

                only_fraud = st.checkbox("Show Only Fraud Transactions")

                if only_fraud:
                    st.dataframe(
                        df[df["Prediction"]=="🚨 Fraud"]
                    )
                else:
                    st.dataframe(df)
                                    # ---------------- CHARTS ----------------

                st.markdown("---")
                st.subheader("📊 Fraud Analysis Dashboard")

                col1, col2 = st.columns(2)

                with col1:

                    chart = pd.DataFrame({
                        "Type": ["Fraud", "Genuine"],
                        "Count": [fraud, genuine]
                    })

                    fig = px.pie(
                        chart,
                        values="Count",
                        names="Type",
                        title="Fraud vs Genuine Transactions",
                        hole=0.4
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:

                    fig2 = px.histogram(
                        df,
                        x="Fraud Probability",
                        nbins=20,
                        title="Fraud Probability Distribution"
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                # ---------------- DOWNLOAD ----------------

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇ Download Prediction Report",
                    data=csv,
                    file_name="fraud_prediction.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(e)
        
        # -------------------- ABOUT --------------------

elif menu == "ℹ About":

    st.title("ℹ Credit Card Fraud Detection")

    st.markdown("""
### 📌 Project Overview

This project detects fraudulent credit card transactions using
**Machine Learning**.

### 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Logistic Regression
- Pandas
- NumPy
- Plotly

### 📂 Dataset

- Kaggle Credit Card Fraud Detection Dataset

### 🤖 ML Workflow

1. Data Preprocessing
2. Feature Scaling
3. Logistic Regression Training
4. Prediction
5. Dashboard Visualization

### 👨‍💻 Developer

Built as an end-to-end Machine Learning project for portfolio and resume.
""")

st.markdown("---")

st.caption(
    "💳 Credit Card Fraud Detection Dashboard | Developed using Streamlit & Scikit-Learn"
)
