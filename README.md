AI-based Medical Diagnosis and Training Tool

An AI-powered tool designed to assist in preliminary medical diagnosis and provide interactive training for students and healthcare learners. The repository includes a Flask-based app, example Python scripts, datasets for hypertension/diabetes/lung cancer, and templates/static assets for the web UI. 

---

Overview

This project demonstrates a compact, end-to-end pipeline for medical-support tooling:

- Web application (Flask) to interact with diagnosis models and training modules.  
- Datasets and example scripts to train / evaluate simple ML models using CSV data included in the repo.  
- Templates and static assets for UI pages (forms, result pages, training/quiz pages). 

---

Features

- Preliminary symptom-based diagnosis using trained ML models (example scripts included).  
- Training / quiz modules for learners (disease MCQ dataset present).  
- Datasets included for blood-pressure awareness, diabetes, lung cancer, and MCQs to support model training and quizzes.  
- Simple Flask UI to input patient info and view model output.

---

Repository Structure 

```

AI-based-Medical-Diagnosis-and-Training-Tool/
├── app_flask.py                        # Main Flask application
├── models/                             # Model-related code (preprocessing, saved models)
├── templates/                          # HTML templates for the web UI
├── static/                             # CSS / images / JS assets
├── bp.csv                              # Blood-pressure sample data
├── bp_awareness_dataset_6000.csv       # BP dataset
├── diabetes.csv                        # Diabetes dataset
├── lungcancer.csv                      # Lung cancer dataset
├── disease_mcq_dataset_500.csv         # MCQ dataset for training quizzes
├── p1.py, p2.py, p3.py                 # Example scripts (training / inference / utilities)
└── README.md

````

(Exact file list and datasets are present in the repository.) 

---

Tech Stack

- Backend: Python, Flask  
- Data: CSV datasets (pandas)  
- ML: scikit-learn / any lightweight model code (example scripts included)  
- Frontend: HTML templates + CSS (static folder)  
- Storage: Local CSVs / optionally serialized model files

---

 Getting Started — Local Setup

> Tested with Python 3.8+ (recommended). Install dependencies below before running.

1. Clone the repo
```bash
git clone https://github.com/YesadeSamiksha/AI-based-Medical-Diagnosis-and-Training-Tool.git
cd AI-based-Medical-Diagnosis-and-Training-Tool
````

2. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install likely dependencies
   Create a `requirements.txt` if not present, then:

```bash
pip install flask pandas scikit-learn numpy
# If you use Streamlit for demos:
pip install streamlit
```

> If the repository already contains a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

---

Run the Flask App

Start the web app (the repository contains `app_flask.py` — adjust name if different):

```bash
python app_flask.py
```

Open your browser at:

```
http://127.0.0.1:5000/
```

If the project uses `FLASK_APP` / `flask run` instead, use:

```bash
export FLASK_APP=app_flask.py       # (Linux / macOS)
set FLASK_APP=app_flask.py          # (Windows CMD)
flask run
```

---

Example: Run model scripts

There are example Python scripts (`p1.py`, `p2.py`, `p3.py`) — they typically contain training, inference, or data-prep routines. To run an example:

```bash
python p1.py
```

Check inside the scripts to see required arguments or dataset file paths.

---

Datasets & Data Notes

The repo includes multiple CSV files for different conditions:

* `bp.csv`, `bp_awareness_dataset_6000.csv` — blood pressure data
* `diabetes.csv` — diabetes dataset
* `lungcancer.csv` — lung cancer dataset
* `disease_mcq_dataset_500.csv` — MCQ set for training / quizzes

These are intended for learning, prototyping, and demo purposes. Before using in production, ensure data provenance, ethical review, and compliance with privacy regulations.

---

Important: Not for Clinical Use

This tool is intended for educational, prototyping, and training purposes only. It is **not** a substitute for professional medical diagnosis, treatment, or advice. Do not use for clinical decision-making without rigorous validation and regulatory approvals.

---

Roadmap

* Improve model accuracy with feature engineering and hyperparameter tuning.
* Add a model registry & save trained models (joblib / pickle).
* Add REST API endpoints for model inference.
* Add authentication & role-based access (student / instructor / admin).
* Add unit tests, CI workflow, and Dockerfile for reproducible deployment.


---

Contributing

Contributions are welcome — open an issue to discuss features or submit pull requests. When contributing:

* Add tests for new utilities / model code.
* Document dataset preprocessing and model training steps.
* Keep sensitive data out of the repo.

---

Author

Yesade Samiksha — developer & maintainer.
For questions or collaboration, open an issue on the repo.



---
