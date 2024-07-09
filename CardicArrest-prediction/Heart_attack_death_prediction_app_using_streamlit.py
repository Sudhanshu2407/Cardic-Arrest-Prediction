import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


# Set the title and the style of the web app
st.set_page_config(page_title="❤️ Cardiac Arrest Death Prediction", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        # background-color: #f5f5f5;
        color: #333;
    }
    .main {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #ff4d4d;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #ff4d4d;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4d4d;
        color: #ffffff;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\sudhanshu_projects\project-task-training-course\CardicArrest-prediction\heart_failure_dataset.csv')

# Preprocess data
def preprocess_data(df):
    df.drop("time",axis=1,inplace=True)
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoking'] = le.fit_transform(df['smoking'])
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Display visualizations
def display_visualizations(df):
    st.subheader("Exploratory Data Analysis")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, x='DEATH_EVENT', ax=axs[0])
    axs[0].set_title('Death Event Count')
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Feature Correlation')
    st.pyplot(fig)

# Main function
def main():
    st.title("❤️ Cardiac Arrest Death Prediction")
    st.sidebar.title("Cardiac Arrest Death Prediction")
    st.sidebar.markdown("Predict the likelihood of death from cardiac arrest based on patient data.")

    df = load_data()

    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Raw Data")
        st.write(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    if st.sidebar.checkbox("Show Data Visualizations", False):
        display_visualizations(df)

    if st.sidebar.button("Train Model"):
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
        st.subheader("Classification Report")
        st.write(classification_report(y_test, y_pred))

    st.sidebar.subheader("Predict Cardiac Arrest Death")
    age = st.sidebar.slider("Age", 0, 100, 50)
    anaemia = st.sidebar.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.sidebar.slider("Creatinine Phosphokinase", 0, 1000, 250)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.sidebar.slider("Ejection Fraction", 10, 80, 35)
    high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0, 1])
    platelets = st.sidebar.slider("Platelets (kiloplatelets/mL)", 0.0, 800.0, 250.0)
    serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.0, 10.0, 1.0)
    serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 100, 150, 135)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    smoking = st.sidebar.selectbox("Smoking", [0, 1])

    sex_n=0 if sex=="Male" else 1
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex_n, smoking]])
    
    import pickle 
    
    model = pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\CardicArrest-prediction\heart_failure_prediction.pkl","rb"))
    
    if st.sidebar.button("Predict"):
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
       
        st.subheader("Prediction result is")
        st.write("Predicted Cardiac Arrest Death: ")
        result="Yes" if prediction[0]==1 else "No"
        st.write(result)
    
    
if __name__ == "__main__":
    main()
