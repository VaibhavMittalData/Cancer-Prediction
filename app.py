import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 📌 Load Dataset & Preprocess
# ============================

@st.cache_data
def load_data():
    df = pd.read_csv(r"processed_lung_cancer.csv")
    return df

df = load_data()

@st.cache_resource
def preprocess_and_train():
    data = df.copy()
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop('lung_cancer', axis=1)
    y = data['lung_cancer']
    
    scaler = StandardScaler()
    X[['age', 'pack_years']] = scaler.fit_transform(X[['age', 'pack_years']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }

    trained_models = {}
    acc_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        predictions = model.predict(X_test)
        acc_scores[name] = accuracy_score(y_test, predictions)

    return trained_models, acc_scores, label_encoders, scaler, X_test, y_test

models, acc_metrics, label_encoders, scaler, X_test, y_test = preprocess_and_train()


# ============================
# 📌 STREAMLIT UI
# ============================

st.sidebar.title("🩺 Lung Cancer Prediction")
page = st.sidebar.radio("Navigate", ["📊 Dataset Overview", "🔍 EDA", "🤖 Model Performance", "🩺 Prediction Tool"])

# ============================
# 📌 PAGE 1 — Dataset Overview
# ============================

if page == "📊 Dataset Overview":
    st.title("📊 Lung Cancer Dataset Overview")
    st.write(df.head())
    st.write("### 🔢 Dataset Shape:", df.shape)
    st.write("### 📌 Gender Distribution")
    st.bar_chart(df['gender'].value_counts())
    st.write("### 📌 Second Hand Smoke Exposure")
    st.bar_chart(df['secondhand_smoke_exposure'].value_counts())
    st.write()
    st.write("### 📌 Target Distribution")
    st.bar_chart(df['lung_cancer'].value_counts())


    

# ============================
# 📌 PAGE 2 — EDA
# ============================

elif page == "🔍 EDA":
    st.title("🔍 Lung Cancer Risk Factors (EDA)")
    st.write("Statistical summary")
    st.table(df.describe())

    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Pack Years Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(df['pack_years'], ax=ax)
    st.pyplot(fig)

    col = st.selectbox("📌 Select feature against Lung Cancer", 
                       list(df.columns[:-1]))
    st.write(f"### {col} vs Lung Cancer")
    fig, ax = plt.subplots()
    ax.set_xticklabels([])      # 🔥 Removes labels
    sns.countplot(x=df[col], hue=df['lung_cancer'],ax=ax)
    st.pyplot(fig)

    st.write("### 📌 Correlation Heatmap")

    # Convert categorical columns to numeric codes
    corr_df = df.copy()
    for col in corr_df.select_dtypes(include=['object']).columns:
        corr_df[col] = corr_df[col].astype('category').cat.codes

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    st.pyplot(fig)




# ============================
# 📌 PAGE 3 — Model Performance
# ============================

elif page == "🤖 Model Performance":
    st.title("🤖 Model Comparison & Performance")

    st.write("### 📌 Accuracy of All Models")
    st.bar_chart(acc_metrics)

    model_name = st.selectbox("📌 Select Model to View Details", list(models.keys()))
    model = models[model_name]

    st.subheader(f"📌 Confusion Matrix — {model_name}")
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    st.pyplot(fig)

    st.write("### 📌 Classification Report")
    y_test = label_encoders["lung_cancer"].inverse_transform(y_test)
    pred = label_encoders["lung_cancer"].inverse_transform(pred)
    report = classification_report(y_test, pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.table(df_report)


# ============================
# 📌 PAGE 4 — Prediction Tool
# ============================


elif page == "🩺 Prediction Tool":
    st.title("🩺 Lung Cancer Risk Prediction")

    selected_model = st.sidebar.selectbox("🤖 Select Model for Prediction", list(models.keys()))
    model = models[selected_model]

    st.write("### 📌 Enter Patient Details")

    age = st.number_input("Age", 18, 100)
    gender = st.radio("Gender", ["Male", "Female"])
    smoke = st.radio("Do you smoke?", ["Yes", "No"])

    if smoke == "Yes":
        cpd = st.number_input("Cigarettes per day", 1, 60)
        yrs = st.number_input("Years Smoked", 1, 60)
        pack_years = round((cpd * yrs) / 20, 2)
        st.write(f"📌 **Pack Years Calculated:** {pack_years}")
        secondhand = st.radio("Exposure to Smoke Around You?", ["Yes", "No"])
    else:
        pack_years = 0
        secondhand = "No"

    pollution = st.selectbox("Pollution Exposure", ["Low", "Medium", "High"])
    chemical = st.radio("Chemical Exposure", ["Yes", "No"])
    copd = st.radio("Breathing Problems (COPD)", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Not drinking/Light", "Moderate", "Heavy"])
    family = st.radio("Family History of Lung Cancer", ["Yes", "No"])

    if st.button("🔮 Predict"):

        # 🔧 Fix unseen label issue
        if alcohol == "Not drinking/Light":
            alcohol = "Not drinking"

        # Build user input DataFrame
        user_data = pd.DataFrame(
            [[age, gender, pack_years, pollution, chemical, secondhand, copd, alcohol, family]],
            columns=df.drop('lung_cancer', axis=1).columns
        )

        # Encode categorical values
        for col in user_data.columns:
            if col in label_encoders:
                user_data[col] = label_encoders[col].transform(user_data[col])

        # Scale numerical values
        user_data[['age', 'pack_years']] = scaler.transform(user_data[['age', 'pack_years']])

        # Prediction
        prediction = model.predict(user_data)[0]
        result = label_encoders['lung_cancer'].inverse_transform([prediction])[0]

        if result == "Yes":
            st.error("🚨 High Risk of Lung Cancer Detected!")
        else:
            st.success("🟢 No Significant Risk Detected")

        st.info(f"📌 Model Accuracy: **{acc_metrics[selected_model]*100:.2f}%**")
