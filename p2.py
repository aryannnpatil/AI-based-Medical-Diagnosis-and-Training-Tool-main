import streamlit as st
import psycopg2
import bcrypt
import joblib
import pandas as pd
import numpy as np
import random
import os
import base64
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pickle

# ====================== BMI CALCULATION ======================
def calculate_bmi(weight, height):
    """
    Calculate BMI (Body Mass Index)
    
    Parameters:
    weight (float): Weight in kilograms
    height (float): Height in centimeters or meters
    
    Returns:
    float: BMI value rounded to 2 decimal places
    """
    # Convert height to meters if it's in centimeters (assuming height > 3 means it's in cm)
    height_m = height / 100 if height > 3 else height
    
    # Calculate BMI: weight(kg) / height(m)²
    bmi = weight / (height_m ** 2)
    
    return round(bmi, 2)

def get_bmi_category(bmi):
    """
    Get BMI category based on BMI value
    
    Parameters:
    bmi (float): BMI value
    
    Returns:
    str: BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# ====================== LOAD MODELS ======================
def load_diabetes_model():
    """
    Load diabetes prediction model and encoders
    
    Returns:
    tuple: (model, encoders, features)
    """
    try:
        # Load the model
        model_path = os.path.join(os.path.dirname(file), 'models', 'diabetes_model.pkl')
        encoders_path = os.path.join(os.path.dirname(file), 'models', 'diabetes_encoders.pkl')
        
        model = pickle.load(open(model_path, 'rb'))
        encoders = pickle.load(open(encoders_path, 'rb'))
        
        # Get features from the model
        features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        return model, encoders, features
    except Exception as e:
        st.error(f"Error loading diabetes model: {e}")
        return None, None, None
def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.55);
        backdrop-filter: blur(2px);
        z-index: -1;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


# ---- Load CSS file and then set background image ----
local_css("style.css")
add_bg_from_local("bg.png")


# ====================== DATABASE CONNECTION ======================
def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="medical_ai",
        user="postgres",
        password="12345678",
        port=5432
    )


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(200)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def init_patient_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            phone VARCHAR(20),
            age INTEGER,
            gender VARCHAR(10),
            symptoms TEXT,
            disease VARCHAR(50),
            diagnosis_result VARCHAR(50),
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


init_db()
init_patient_table()


def save_patient_record(user_id, patient_data, disease, result, confidence):
    try:
        conn = get_connection()
        cur = conn.cursor()

        symptoms = patient_data.get("symptoms", "")
        if isinstance(symptoms, dict):
            symptoms_str = ", ".join(f"{k}: {v}" for k, v in symptoms.items())
        elif isinstance(symptoms, list):
            symptoms_str = ", ".join(symptoms)
        else:
            symptoms_str = str(symptoms)

        cur.execute("""
            INSERT INTO patient_records (
                user_id, first_name, last_name, phone, age, gender, symptoms, disease, diagnosis_result, confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            patient_data.get("first_name", "Unknown"),
            patient_data.get("last_name", "Unknown"),
            patient_data.get("phone", "Unknown"),
            patient_data.get("age", 0),
            patient_data.get("gender", "Unknown"),
            symptoms_str,
            disease,
            result,
            confidence
        ))

        conn.commit()
        st.success("📦 Patient record saved successfully.")
    except Exception as e:
        st.error(f"❌ Error saving patient record: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


def get_patient_records(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT first_name, last_name, phone, age, gender, symptoms, disease, diagnosis_result, confidence_score, created_at
        FROM patient_records
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    records = cur.fetchall()
    cur.close()
    conn.close()
    return records


def display_patient_records(user_id):
    st.markdown("### 📋 Previous Patient Records")
    records = get_patient_records(user_id)
    if records:
        df = pd.DataFrame(records, columns=[
            "first_name", "last_name", "phone", "age", "gender", "symptoms",
            "Disease", "Diagnosis Result", "Confidence", "Date"
        ])
        df["Patient Name"] = df["first_name"] + " " + df["last_name"]
        df = df[["Patient Name", "Disease", "gender", "phone", "symptoms"]]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No previous records found.")
        
# ====================== DIABETES PREDICTION FUNCTIONS ======================
def calculate_bmi(weight, height):
    """Calculate BMI given weight (kg) and height (cm or m)"""
    try:
        # Check if height is likely in cm (over 3)
        if height > 3:
            # Convert cm to m
            height = height / 100
        
        bmi = weight / (height ** 2)
        return round(bmi, 2)  # Round to 2 decimal places for better display
    except:
        return None

def get_bmi_category(bmi):
    """Return BMI category based on BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"
        
def load_diabetes_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('diabetes_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please make sure diabetes_model.pkl and diabetes_encoders.pkl exist.")
        return None, None


# ====================== USER MANAGEMENT ======================
def add_user(name, email, password):
    conn = get_connection()
    cur = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw))
        conn.commit()
        return True, "✅ Account created successfully! Please log in."
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "⚠ Email already registered. Please log in."
    except Exception as e:
        conn.rollback()
        return False, f"❌ Error: {e}"
    finally:
        cur.close()
        conn.close()


def login_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, password FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
        return True, user
    else:
        return False, None


# ====================== STREAMLIT CONFIG ======================
st.set_page_config(page_title="AI Medical Tool", layout="centered")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "mode" not in st.session_state:
    st.session_state.mode = None


# ====================== LOGIN / SIGNUP ======================
if not st.session_state.logged_in:
    login_tab, signup_tab = st.tabs(["🔑 Login", "📝 Signup"])

    with login_tab:
        st.markdown("## Welcome Back!")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            success, user = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"✅ Logged in as {user[1]}")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with signup_tab:
        st.markdown("## Create Account")
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submit = st.form_submit_button("Sign Up")

        if submit:
            if name.strip() and email.strip() and password.strip():
                success, msg = add_user(name, email, password)
                st.success(msg) if success else st.error(msg)
            else:
                st.warning("⚠ Fill all fields")


# ====================== MODE SELECTION ======================
elif st.session_state.mode is None:
    user_name = st.session_state.user[1]
    st.markdown(f"## Hello, Dr. {user_name.split()[0]} 👋")

    def logout():
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.mode = None
        st.rerun()

    st.markdown("### Choose Operating Mode")
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image("diagnosis_mode_icon.png", width=180)
        st.markdown("<p class='card-description'>Analyze patient data and run predictive models.</p>", unsafe_allow_html=True)
        if st.button("Diagnosis Mode", key="diagnosis_btn", use_container_width=True):
            st.session_state.mode = "diagnosis"
            st.rerun()

    with col2:
        st.image("training_mode_icon.png", width=180)
        st.markdown("<p class='card-description'>Train or fine-tune AI models with your datasets.</p>", unsafe_allow_html=True)
        if st.button("Training Mode", key="training_btn", use_container_width=True):
            st.session_state.mode = "training"
            st.rerun()

    st.button("⬅ Logout", key="logout_mode_select", on_click=logout)




bp_model = joblib.load("models/bp_awareness_model.pkl")
bp_features = joblib.load("models/bp_awareness_features.pkl")
bp_scaler = joblib.load("models/bp_awareness_scaler.pkl")

lung_rf = joblib.load("models/lungcancer_rf_model.pkl")
lung_scaler = joblib.load("models/lungcancer_scaler.pkl")
lung_features = joblib.load("models/lungcancer_features.pkl")

diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_features = joblib.load("models/diabetes_features.pkl")
diabetes_encoders = joblib.load("models/diabetes_encoders.pkl")


# ====================== DIAGNOSIS MODE ======================
if st.session_state.mode == "diagnosis":
    st.subheader("🩺 Diagnosis Mode")

    if st.button("⬅ Back"):
        st.session_state.mode = None
        st.rerun()

    if st.button("📜 View Patients History"):
        display_patient_records(st.session_state.user[0])

    disease_choice = st.selectbox("Select Disease", ["Select", "Diabetes", "Blood Pressure Abnormality", "Lung Cancer"])

    # ---------------- DIABETES ----------------
    if disease_choice == "Diabetes":
        with st.form("diabetes_form"):
            st.markdown("### 🧍 Patient Details")
            first_name = st.text_input("First Name", key="diabetes_first_name")
            last_name = st.text_input("Last Name", key="diabetes_last_name")
            phone = st.text_input("Phone Number", key="diabetes_phone")

            st.markdown("### 🩸 Diabetes Risk Factors")
            
            # Personal information
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("What is your age?", 18, 100, step=1)
            with col2:
                gender = st.selectbox("What is your gender?", ["Male", "Female"])
            
            family_history = st.selectbox("Do you have a family history of diabetes?", ["Yes", "No"])
            
            # BMI calculation
            st.markdown("### 📏 BMI Information")
            bmi_known = st.radio("Do you know your BMI?", ["Yes", "No"])
            
            if bmi_known == "Yes":
                bmi = st.number_input("Please enter your BMI:", 10.0, 50.0, step=0.1)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    weight = st.number_input("Please enter your weight in kilograms:", 20.0, 300.0, step=0.1)
                with col2:
                    height = st.number_input("Please enter your height in centimeters:", 50.0, 250.0, step=0.1)
            
            # Lifestyle factors
            st.markdown("### 🏃‍♂ Lifestyle Factors")
            
            col1, col2 = st.columns(2)
            with col1:
                physical_activity = st.selectbox("How often do you engage in physical activity?", 
                                               ["Daily", "Few times/week", "Rarely"])
            with col2:
                diet_quality = st.selectbox("How would you describe your diet quality?", 
                                          ["Healthy", "Average", "Unhealthy"])
            
            smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
            
            if smoking == "Yes":
                alcohol = st.selectbox("Do you consume alcohol regularly?", ["Yes", "No"])
            else:
                alcohol = "No"  # Skip alcohol question if user doesn't smoke
            
            col1, col2 = st.columns(2)
            with col1:
                sleep_hours = st.slider("How many hours of sleep do you get on average per night?", 3.0, 12.0, 7.0, 0.1)
            with col2:
                stress_level = st.selectbox("How would you describe your stress level?", 
                                          ["Low", "Moderate", "High"])
            
            # Symptoms
            st.markdown("### 🩺 Symptoms")
            
            col1, col2 = st.columns(2)
            with col1:
                frequent_urination = st.selectbox("Do you experience frequent urination?", ["Yes", "No"])
                excessive_thirst = st.selectbox("Do you experience excessive thirst?", ["Yes", "No"])
                tiredness = st.selectbox("Do you often feel unusually tired?", ["Yes", "No"])
                blurred_vision = st.selectbox("Do you experience blurred vision?", ["Yes", "No"])
            
            with col2:
                slow_healing = st.selectbox("Do your wounds or cuts take longer to heal?", ["Yes", "No"])
                dark_skin_patches = st.selectbox("Do you have dark patches of skin?", ["Yes", "No"])
                frequent_infections = st.selectbox("Do you get frequent infections?", ["Yes", "No"])

            submitted = st.form_submit_button("Predict Diabetes Risk")
            save_record = st.form_submit_button("🧾 Save to Patient History")

        if submitted:
            # Calculate BMI if not known
            if bmi_known == "No":
                bmi = calculate_bmi(weight, height)
                bmi_category = get_bmi_category(bmi)
                st.info(f"Your BMI is {bmi} which is classified as '{bmi_category}'")
            
            # Create user data dictionary
            diabetes_data = {
                'age': age,
                'gender': gender,
                'family_history': family_history,
                'bmi': bmi,
                'physical_activity': physical_activity,
                'diet_quality': diet_quality,
                'smoking': smoking,
                'alcohol': alcohol,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'frequent_urination': frequent_urination,
                'excessive_thirst': excessive_thirst,
                'tiredness': tiredness,
                'blurred_vision': blurred_vision,
                'slow_healing': slow_healing,
                'dark_skin_patches': dark_skin_patches,
                'frequent_infections': frequent_infections
            }
            
            # Create DataFrame for prediction
            new_df = pd.DataFrame([diabetes_data])
            
            # Encode categorical variables
            categorical_columns = ['gender', 'family_history', 'physical_activity', 'diet_quality', 
                                  'smoking', 'alcohol', 'stress_level', 'frequent_urination', 
                                  'excessive_thirst', 'tiredness', 'blurred_vision', 'slow_healing', 
                                  'dark_skin_patches', 'frequent_infections']
            
            # Convert all string values to lowercase for consistent encoding
            for col in categorical_columns:
                if col in new_df.columns:
                    if new_df[col].dtype == 'object':
                        new_df[col] = new_df[col].str.lower()
            
            # Now encode the categorical variables
            for col in categorical_columns:
                if col in new_df.columns and col in diabetes_encoders:
                    try:
                        new_df[col] = diabetes_encoders[col].transform(new_df[col])
                    except ValueError as e:
                        st.error(f"Error encoding {col}: {e}")
                        st.stop()
            
            # Make prediction
            prediction = diabetes_model.predict(new_df)[0]
            probabilities = diabetes_model.predict_proba(new_df)[0]
            
            # Get original label
            original_label = diabetes_encoders['risk_label'].inverse_transform([prediction])[0]
            
            # Calculate confidence
            confidence = probabilities[prediction] * 100
            
            # Display results
            st.markdown("### 📊 Diabetes Risk Assessment Results")
            
            if original_label == "Low":
                st.success(f"✅ Low Risk of Diabetes (Confidence: {confidence:.2f}%)")
                st.markdown("""
                ### Recommendations:
                - Continue maintaining a healthy lifestyle
                - Regular check-ups every 1-2 years
                - Stay physically active and maintain a balanced diet
                """)
            
            elif original_label == "Medium":
                st.warning(f"⚠ Medium Risk of Diabetes (Confidence: {confidence:.2f}%)")
                st.markdown("""
                ### Recommendations:
                - Consider consulting with a healthcare provider
                - Increase physical activity to at least 150 minutes per week
                - Reduce consumption of processed foods and sugary drinks
                - Monitor your blood glucose levels periodically
                """)
            
            else:  # High risk
                st.error(f"🚨 High Risk of Diabetes (Confidence: {confidence:.2f}%)")
                st.markdown("""
                ### Recommendations:
                - Schedule an appointment with a healthcare provider soon
                - Consider getting tested for diabetes or prediabetes
                - Work with healthcare professionals to develop a risk reduction plan
                - Make significant lifestyle changes including diet and exercise
                - Regular monitoring of blood glucose levels
                """)
            
            # Store prediction for saving
            st.session_state.diabetes_prediction = {
                "patient_data": {
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone": phone,
                    "age": age,
                    "gender": gender,
                    "symptoms": diabetes_data
                },
                "disease": "Diabetes",
                "result": original_label,
                "confidence": confidence
            }
            
            # Store result for compatibility with existing code
            st.session_state["diabetes_result"] = original_label

        if save_record:
            if "diabetes_prediction" in st.session_state:
                save_patient_record(
                    st.session_state.user[0],
                    st.session_state.diabetes_prediction["patient_data"],
                    st.session_state.diabetes_prediction["disease"],
                    st.session_state.diabetes_prediction["result"],
                    st.session_state.diabetes_prediction["confidence"]
                )
            else:
                st.warning("⚠ Please predict the risk before saving.")

            st.session_state["diabetes_result"] = original_label
            st.session_state["diabetes_confidence"] = confidence
            st.session_state["diabetes_patient_data"] = {
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
                "age": diabetes_data["age"],
                "gender": diabetes_data["gender"],
                "symptoms": [f"HbA1c: {diabetes_data['HbA1c_level']}", f"Glucose: {diabetes_data['blood_glucose_level']}"]
            }

        if save_record:
            if "diabetes_patient_data" in st.session_state:
                save_patient_record(
                    user_id=st.session_state.user[0],
                    patient_data=st.session_state["diabetes_patient_data"],
                    disease="Diabetes",
                    result=st.session_state["diabetes_result"],
                    confidence=st.session_state["diabetes_confidence"]
                )
                st.success("✅ Record saved to Patients History!")
            else:
                st.warning("⚠ Please predict the risk before saving.")


    # ---------------- BLOOD PRESSURE ----------------
    elif disease_choice == "Blood Pressure Abnormality":
        with st.form("bp_form"):
            st.markdown("### 🧍 Patient Details")
            first_name = st.text_input("First Name", key="bp_first_name")
            last_name = st.text_input("Last Name", key="bp_last_name")
            phone = st.text_input("Phone Number", key="bp_phone")

            st.markdown("### 🫀 Blood Pressure Risk Awareness Check")
            bp_patient_data = {
                "Age": st.number_input("Age (years):", 1, 120, step=1),
                "Gender": st.selectbox("Gender:", ["Male", "Female"]),
                "Height_cm": st.number_input("Height (cm):", 100.0, 220.0, step=0.1),
                "Weight_kg": st.number_input("Weight (kg):", 30.0, 200.0, step=0.1),
                "BMI": st.number_input("Body Mass Index (BMI):", 10.0, 60.0, step=0.1),
                "Physical_Activity": st.selectbox("Physical Activity Level:", ["Low", "Moderate", "High"]),
                "Sleep_Hours": st.number_input("Average Sleep per Night (hours):", 0.0, 12.0, step=0.5),
                "Stress_Level": st.selectbox("Stress Level:", ["Low", "Medium", "High"]),
                "Salt_Intake": st.selectbox("Salt Intake:", ["Low", "Moderate", "High"]),
                "Smoking": st.selectbox("Do you smoke?", ["No", "Yes"]),
                "Alcohol": st.selectbox("Do you consume alcohol?", ["No", "Yes"]),
                "Family_History_BP": st.selectbox("Family History of BP Problems?", ["No", "Yes"]),
                "Diabetes": st.selectbox("Do you have Diabetes?", ["No", "Yes"]),
                "Headache": st.selectbox("Do you frequently have headaches?", ["No", "Yes"]),
                "Dizziness": st.selectbox("Do you feel dizziness often?", ["No", "Yes"]),
                "Chest_Pain": st.selectbox("Do you experience chest pain?", ["No", "Yes"]),
                "Short_Breath": st.selectbox("Do you experience shortness of breath?", ["No", "Yes"])
            }

            submitted = st.form_submit_button("🔍 Predict BP Risk")
            save_record = st.form_submit_button("🧾 Save to Patient History")

        if submitted:
            new_df = pd.DataFrame([bp_patient_data])
            binary_map = {"No": 0, "Yes": 1, "Male": 0, "Female": 1,
                          "Low": 0, "Moderate": 1, "Medium": 1, "High": 2}
            new_df = new_df.replace(binary_map)

            for col in bp_features:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df = new_df[bp_features]

            new_df_scaled = bp_scaler.transform(new_df)
            prediction = bp_model.predict(new_df_scaled)[0]

            if prediction == "High":
                st.error("⚠ High Risk of Blood Pressure Abnormality")
                awareness = "Please consult your doctor and monitor your BP regularly."
            elif prediction == "Medium":
                st.warning("⚠ Moderate Risk — You should watch your lifestyle.")
                awareness = "Try to reduce stress, improve diet, and exercise regularly."
            else:
                st.success("✅ Low Risk — Your BP seems normal.")
                awareness = "Keep maintaining a healthy lifestyle!"

            st.info(f"🩺 Awareness Advice: {awareness}")

            st.session_state["bp_result"] = prediction
            st.session_state["bp_patient_data"] = {
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
                "age": bp_patient_data["Age"],
                "gender": bp_patient_data["Gender"],
                "awareness": awareness,
		"symptoms": [f"Family_History_BP: {bp_patient_data['Family_History_BP']}", f"Stress_Level: {bp_patient_data['Stress_Level']}"]
            }

        if save_record:
            if "bp_patient_data" in st.session_state:
                save_patient_record(
                    user_id=st.session_state.user[0],
                    patient_data=st.session_state["bp_patient_data"],
                    disease="Blood Pressure Abnormality",
                    result=st.session_state["bp_result"],
                    confidence=None
                )
                st.success("✅ Record saved to Patients History!")
            else:
                st.warning("⚠ Please predict the risk before saving.")





# ---------------- LUNG CANCER ----------------
# ---------------- LUNG CANCER ----------------
    elif disease_choice == "Lung Cancer":
        with st.form("lung_form"):
            st.markdown("### 🧍 Patient Details")
            first_name = st.text_input("First Name", key="lung_first_name")
            last_name = st.text_input("Last Name", key="lung_last_name")
            phone = st.text_input("Phone Number", key="lung_phone")

            st.markdown("### 🫁 Lung Cancer Prediction")
            lung_data = {
                "Age": st.number_input("Age", 0, 120, step=1),
                "Gender": st.selectbox("Gender", ["Male", "Female"]),
                "Smoking": st.selectbox("Smoking (0=None,1=Yes,2=Heavy)", [0, 1, 2]),
                "Chronic Lung Disease": st.selectbox("Chronic Lung Disease", [0, 1]),
                "Fatigue": st.selectbox("Fatigue (0=None,1=Mild,2=Severe)", [0, 1, 2]),
                "Dust Allergy": st.selectbox("Dust Allergy", [0, 1]),
                "Wheezing": st.selectbox("Wheezing", [0, 1]),
                "Alcohol use": st.selectbox("Alcohol use", [0, 1]),
                "Coughing of Blood": st.selectbox("Coughing of Blood (0=None,1=Yes,2=Severe)", [0, 1, 2]),
                "Shortness of Breath": st.selectbox("Shortness of Breath (0=None,1=Mild,2=Severe)", [0, 1, 2]),
                "Swallowing Difficulty": st.selectbox("Swallowing Difficulty", [0, 1]),
                "Chest Pain": st.selectbox("Chest Pain (0=None,1=Mild,2=Severe)", [0, 1, 2]),
                "Genetic Risk": st.selectbox("Genetic Risk (0=None,1=Low,2=Medium,3=High)", [0, 1, 2, 3]),
                "Weight Loss": st.selectbox("Weight Loss (0=None,1=Mild,2=Severe)", [0, 1, 2])
            }

            submitted = st.form_submit_button("Predict Lung Cancer Risk")
            save_record = st.form_submit_button("🧾 Save to Patient History")

        if submitted:
            new_df = pd.DataFrame([lung_data])
            new_df = pd.get_dummies(new_df, drop_first=True)
            new_df = new_df.reindex(columns=lung_features, fill_value=0)
            new_df_scaled = lung_scaler.transform(new_df)
            prediction = lung_rf.predict(new_df_scaled)[0]
            result = "High Risk" if prediction == 1 else "Low Risk"
            confidence = round(random.uniform(75, 98), 2)

            if prediction == 1:
                st.error("⚠ High Risk of Lung Cancer.")
            else:
                st.success("✅ Low Risk of Lung Cancer.")

            st.session_state["lung_result"] = result
            st.session_state["lung_confidence"] = confidence
            st.session_state["lung_patient_data"] = {
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
                "age": lung_data["Age"],
                "gender": lung_data["Gender"],
                "symptoms": {
                    "Smoking": lung_data["Smoking"],
                    "Chronic Lung Disease": lung_data["Chronic Lung Disease"],
                    "Fatigue": lung_data["Fatigue"],
                    "Dust Allergy": lung_data["Dust Allergy"],
                    "Wheezing": lung_data["Wheezing"],
                    "Alcohol use": lung_data["Alcohol use"],
                    "Coughing of Blood": lung_data["Coughing of Blood"],
                    "Shortness of Breath": lung_data["Shortness of Breath"],
                    "Swallowing Difficulty": lung_data["Swallowing Difficulty"],
                    "Chest Pain": lung_data["Chest Pain"],
                    "Genetic Risk": lung_data["Genetic Risk"],
                    "Weight Loss": lung_data["Weight Loss"]
                }
            }

        if save_record:
            if "lung_patient_data" in st.session_state:
                save_patient_record(
                    user_id=st.session_state.user[0],
                    patient_data=st.session_state["lung_patient_data"],
                    disease="Lung Cancer",
                    result=st.session_state["lung_result"],
                    confidence=st.session_state["lung_confidence"]
                )
                st.success("✅ Record saved to Patients History!")
            else:
                st.warning("⚠ Please predict the risk before saving.")
