# AI-Based Medical Diagnosis and Training Tool - Project Summary

## 📋 Project Overview

This is a comprehensive web-based medical application that combines AI-powered disease diagnosis with an interactive training/quiz system for medical students and healthcare learners. The application uses machine learning models to predict health risks and provides an educational platform for medical knowledge assessment.

---

## 🛠️ Technologies & Frameworks Used

### Backend
- **Flask** (v2.0+) - Python web framework for building the application
- **SQLite3** - Lightweight database for storing user accounts and patient records
- **Bcrypt** - Password hashing and authentication security
- **Pickle/Joblib** - Model serialization and loading

### Machine Learning & Data Science
- **Scikit-learn** (v1.0+) - Machine learning library for model training and predictions
- **Pandas** (v1.3+) - Data manipulation and CSV processing
- **NumPy** (v1.20+) - Numerical computations
- **Joblib** (v1.1+) - Efficient model persistence

### Frontend
- **HTML5/CSS3** - Templates with inline styling
- **Jinja2** - Template engine (built into Flask)
- **Responsive Design** - Glass morphism UI with gradient effects

---

## 🤖 Machine Learning Models

### 1. **Diabetes Risk Prediction Model**
- **File**: `models/diabetes_model.pkl`
- **Algorithm**: Random Forest Classifier
- **Features**: 17 input features including:
  - Demographics: age, gender, family history
  - Lifestyle: BMI, physical activity, diet quality, smoking, alcohol, sleep hours, stress level
  - Symptoms: frequent urination, excessive thirst, tiredness, blurred vision, slow healing, dark skin patches, frequent infections
- **Output**: Risk classification (Low, Medium, High)
- **Additional Files**:
  - `diabetes_features.pkl` - Feature names
  - `diabetes_encoders.pkl` - Label encoders for categorical variables
  - `diabetes_imputer.pkl` - Imputer for handling missing values
- **Training Script**: `p1.py`
- **Dataset**: `diabetes.csv` (risk score and risk label based)

### 2. **Blood Pressure Risk Assessment Model**
- **File**: `models/bp_awareness_model.pkl` (primary) or `models/bp_model.pkl`
- **Algorithm**: Classification model (likely Random Forest or Logistic Regression)
- **Features**: 13+ input features including:
  - Demographics: age, gender, height, weight, BMI
  - Lifestyle: physical activity, sleep hours, stress level, salt intake, smoking, alcohol
  - Medical History: family history of BP, diabetes
  - Symptoms: headache, dizziness, chest pain, shortness of breath
- **Output**: BP Risk Level (Low, Medium, High)
- **Additional Files**:
  - `bp_awareness_features.pkl` / `bp_features.pkl` - Feature columns
  - `bp_awareness_scaler.pkl` / `bp_scaler.pkl` - StandardScaler for feature normalization
- **Dataset**: `bp_awareness_dataset_6000.csv` (6000 records) or `bp.csv`

### 3. **Lung Cancer Risk Prediction Model**
- **File**: `models/lungcancer_rf_model.pkl`
- **Algorithm**: Random Forest Classifier
- **Features**: 13 input features including:
  - Demographics: age, gender
  - Risk Factors: smoking level (0-2), chronic lung disease, genetic risk (0-3)
  - Symptoms: fatigue (0-2), dust allergy, wheezing, alcohol use, coughing blood (0-2), shortness of breath (0-2), swallowing difficulty, chest pain (0-2), weight loss (0-2)
- **Output**: Cancer Risk Level (Low, Medium, High)
- **Additional Files**:
  - `lungcancer_features.pkl` - Feature names after one-hot encoding
  - `lungcancer_scaler.pkl` - StandardScaler for normalization
  - `lungcancer_logreg_model.pkl` - Alternate Logistic Regression model
- **Training Script**: `p3.py`
- **Dataset**: `lungcancer.csv`

---

## 📁 Project Structure

```
AI-based-Medical-Diagnosis-and-Training-Tool-main/
│
├── app_flask.py                    # Main Flask application (1220 lines)
├── requirements.txt                # Python dependencies
├── medical_ai.db                   # SQLite database (auto-created)
│
├── models/                         # Pre-trained ML models
│   ├── diabetes_model.pkl
│   ├── diabetes_features.pkl
│   ├── diabetes_encoders.pkl
│   ├── diabetes_imputer.pkl
│   ├── bp_awareness_model.pkl
│   ├── bp_awareness_features.pkl
│   ├── bp_awareness_scaler.pkl
│   ├── lungcancer_rf_model.pkl
│   ├── lungcancer_features.pkl
│   └── lungcancer_scaler.pkl
│
├── templates/                      # HTML templates (Jinja2)
│   ├── base.html                   # Base template with common layout
│   ├── login.html                  # Login/Signup page
│   ├── mode_selection.html         # Choose Diagnosis or Training mode
│   ├── diagnosis.html              # Disease selection page
│   ├── diagnosis_diabetes.html     # Diabetes risk assessment form
│   ├── diagnosis_bp.html           # Blood pressure assessment form
│   ├── diagnosis_lung.html         # Lung cancer assessment form
│   ├── training_dashboard.html     # Training mode dashboard
│   ├── quiz_levels.html            # Select quiz difficulty
│   ├── quiz_name_input.html        # Enter name before quiz
│   ├── quiz.html                   # Quiz interface with questions
│   ├── quiz_result.html            # Quiz results display
│   ├── training_progress.html      # User's quiz history
│   └── training_leaderboard.html   # Top performers leaderboard
│
├── static/                         # Static assets
│   ├── style.css                   # Main stylesheet
│   ├── bg.png                      # Background image
│   ├── bgg.png                     # Alternative background
│   ├── diagnosis_mode_icon.png     # Diagnosis mode icon
│   ├── training_mode_icon.png      # Training mode icon
│   ├── easy.png                    # Easy level icon
│   ├── moderate.png                # Moderate level icon
│   ├── hard.png                    # Hard level icon
│   ├── gotoquiz.png               # Quiz navigation icon
│   ├── leaderboard.png            # Leaderboard icon
│   └── myprogress.png             # Progress tracking icon
│
├── Datasets/                       # Training datasets
│   ├── diabetes.csv                # Diabetes training data
│   ├── bp_awareness_dataset_6000.csv  # BP training data (6000 records)
│   ├── bp.csv                      # Alternative BP dataset
│   ├── lungcancer.csv              # Lung cancer training data
│   └── disease_mcq_dataset_500.csv # Quiz questions (500 MCQs)
│
├── Training Scripts/
│   ├── p1.py                       # Diabetes model training script
│   ├── p2.py                       # Streamlit alternative (not used)
│   └── p3.py                       # Lung cancer model training script
│
└── README.md                       # Original project documentation
```

---

## 🎯 Key Features

### 1. **User Authentication System**
- Secure signup and login
- Password hashing with Bcrypt
- Session-based authentication
- SQLite database for user storage

### 2. **Diagnosis Mode**
Three AI-powered disease risk assessments:

#### A. Diabetes Risk Assessment
- Comprehensive 17-parameter evaluation
- BMI calculation (manual or auto-calculated from height/weight)
- Lifestyle and symptom analysis
- Confidence score with recommendations
- Risk levels: Low, Medium, High

#### B. Blood Pressure Risk Assessment
- 13+ parameter health evaluation
- Physical and lifestyle factors
- Family history consideration
- Personalized awareness advice
- Risk levels: Low, Medium, High

#### C. Lung Cancer Risk Assessment
- 13 symptom and risk factor evaluation
- Numeric severity scales (0-3)
- Genetic risk assessment
- Detailed risk classification
- Risk levels: Low, Medium, High

**Common Features**:
- Patient information capture (name, phone, age, gender)
- Save diagnosis to patient history
- View historical records
- Recommendations based on risk level

### 3. **Training Mode**
Interactive quiz system for medical education:

#### Quiz System
- **Three Difficulty Levels**: Easy, Moderate, Hard
- **Question Format**: Multiple choice (4 options)
- **Question Pool**: 500 disease-related MCQs from dataset
- **Features**:
  - 10 random questions per quiz
  - Immediate feedback after each answer
  - Drug recommendations for each condition
  - Detailed explanations
  - Real-time score tracking
  - Progress bar visualization

#### Progress Tracking
- Quiz history with date/time stamps
- Score and percentage tracking
- Difficulty level record
- Personal performance analytics

#### Leaderboard System
- Top 10 performers display
- Score-based ranking
- Difficulty level shown
- Competition tracking

---

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
```

### Patient Records Table
```sql
CREATE TABLE patient_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    first_name TEXT,
    last_name TEXT,
    phone TEXT,
    age INTEGER,
    gender TEXT,
    symptoms TEXT,
    disease TEXT,
    diagnosis_result TEXT,
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Clone or Navigate to Project Directory**
   ```bash
   cd AI-based-Medical-Diagnosis-and-Training-Tool-main
   ```

2. **Create Virtual Environment** (if not exists)
   ```bash
   python -m venv .venv
   ```

3. **Activate Virtual Environment**
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**
   ```bash
   python app_flask.py
   ```

6. **Access the Application**
   - Open browser and go to: `http://127.0.0.1:5000`
   - Create an account and start using the app

---

## 🧪 Model Training (Optional)

If you want to retrain the models with updated data:

### Train Diabetes Model
```bash
python p1.py
```
- Reads: `diabetes.csv`
- Outputs: Models to `models/` folder

### Train Lung Cancer Model
```bash
python p3.py
```
- Reads: `lungcancer.csv`
- Outputs: Models to `models/` folder

**Note**: Blood pressure model training script is not included but follows similar pattern.

---

## 📊 Data Flow

### Diagnosis Flow
1. User logs in → Selects Diagnosis Mode
2. Chooses disease type (Diabetes/BP/Lung Cancer)
3. Fills patient information form
4. Enters health metrics and symptoms
5. App processes data:
   - Validates inputs
   - Encodes categorical variables
   - Scales numerical features
   - Loads appropriate ML model
   - Makes prediction
6. Displays results with:
   - Risk level
   - Confidence score (if available)
   - Personalized recommendations
7. Option to save to patient history

### Training/Quiz Flow
1. User logs in → Selects Training Mode
2. Chooses quiz difficulty level
3. Enters name for tracking
4. System randomly selects 10 questions from dataset
5. For each question:
   - Display question with 4 options
   - User submits answer
   - Immediate feedback (correct/wrong)
   - Show explanation and drug information
6. Calculate final score and percentage
7. Save to progress history
8. Update leaderboard if top score
9. Display results with performance comment

---

## 🎨 UI/UX Features

- **Glass Morphism Design**: Modern translucent effects with backdrop blur
- **Gradient Accents**: Cyan (#00ffd5) primary color scheme
- **Responsive Layout**: Works on different screen sizes
- **Visual Feedback**: Color-coded results (green=low, yellow=medium, red=high)
- **Progress Indicators**: Visual progress bars in quizzes
- **Icon Integration**: Emoji-based icons for better UX
- **Background Images**: Aesthetic medical-themed backgrounds

---

## 🔐 Security Features

- Password hashing with Bcrypt (salt rounds)
- Session-based authentication
- SQL injection prevention (parameterized queries)
- Input validation on all forms
- Unique email constraint for users

---

## ⚠️ Important Notes

1. **Development Server**: Currently using Flask's built-in server (debug mode). For production, use a WSGI server like Gunicorn or uWSGI.

2. **Secret Key**: Change `app.secret_key` in production to a strong random key.

3. **Database**: SQLite is suitable for development/small deployments. For production with multiple concurrent users, consider PostgreSQL or MySQL.

4. **Model Updates**: Retrain models periodically with new data to maintain accuracy.

5. **Data Privacy**: Ensure compliance with healthcare data regulations (HIPAA, GDPR) if deploying with real patient data.

---

## 📈 Potential Improvements

- Add more disease prediction models (heart disease, cancer types)
- Implement data visualization for patient trends
- Add PDF report generation for diagnosis results
- Implement email notifications for high-risk cases
- Add admin panel for user management
- Enhance quiz with timed challenges
- Add multi-language support
- Implement API endpoints for mobile app integration
- Add model explainability (SHAP/LIME values)

---

## 👨‍💻 Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Client)                      │
│                   HTML/CSS/JavaScript                    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Flask Application                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Routes & Controllers (app_flask.py)             │  │
│  │  - Authentication                                 │  │
│  │  - Diagnosis endpoints                           │  │
│  │  - Training/Quiz endpoints                       │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────┴──────────────────────────┐  │
│  │     Business Logic                               │  │
│  │  - Form validation                               │  │
│  │  - Data preprocessing                            │  │
│  │  - Model inference                               │  │
│  └──────────────────────────────────────────────────┘  │
└──────────┬────────────────────────┬────────────────────┘
           │                        │
           ▼                        ▼
┌─────────────────────┐  ┌─────────────────────────────┐
│  SQLite Database    │  │   ML Models (Joblib/Pickle) │
│  - Users            │  │   - Diabetes RF             │
│  - Patient Records  │  │   - BP Classifier           │
└─────────────────────┘  │   - Lung Cancer RF          │
                         │   - Scalers & Encoders      │
                         └─────────────────────────────┘
```

---

## 📝 License & Credits

This project demonstrates the integration of machine learning with web applications for healthcare education and preliminary diagnosis support. It should be used for educational purposes only and not as a replacement for professional medical advice.

**Last Updated**: February 2026
**Python Version**: 3.10+
**Flask Version**: 2.0+
**Status**: Development/Educational

---

*For questions or contributions, refer to the original repository or contact the development team.*
