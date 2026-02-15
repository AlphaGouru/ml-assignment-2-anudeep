import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Classification Models Demo")

uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=';')
    model = joblib.load(f"models/{model_name.replace(' ', '_')}.pkl")

    X = data.drop("y", axis=1)
    y = data["y"]

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
