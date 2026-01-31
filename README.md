# Student Exam Performance Prediction – ML Project

This project predicts a student’s Maths score based on demographic and academic inputs using a trained Machine Learning model and a Flask web application.

It demonstrates an end-to-end ML workflow:
- Data ingestion and preprocessing
- Model training and evaluation
- Model persistence (artifacts)
- Prediction pipeline
- Flask-based UI for user interaction


## Project Overview

The goal of this project is to predict Maths performance of students using features such as:
- Gender
- Race / Ethnicity
- Parental education level
- Lunch type
- Test preparation course
- Reading score
- Writing score

The trained model and preprocessing pipeline are saved as artifacts and reused during prediction.


## Tech Stack

- Programming Language: Python
- Web Framework: Flask
- Machine Learning: scikit-learn
- Data Handling: pandas, numpy
- Model Persistence: pickle
- Logging & Exception Handling: Custom logger and exception classes


## Project Structure

```
ML_Project/
│
├── app.py                         # Flask application
├── artifacts/
│   ├── model.pkl                  # Trained ML model
│   └── preprocessor.pkl           # Data preprocessing pipeline
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── utils.py                   # Utility functions (save/load objects)
│   ├── exception.py               # Custom exception handling
│   └── logger.py                  # Logging configuration
│
├── templates/
│   └── home.html                  # Flask UI template
│
├── requirements.txt
└── README.md
```



## Training Pipeline (High Level)

1. Data Ingestion
   - Reads raw dataset
   - Splits data into train and test sets

2. Data Transformation
   - Handles categorical and numerical features
   - Applies encoding and scaling using pipelines
   - Saves preprocessing object as preprocessor.pkl

3. Model Training
   - Trains multiple regression models
   - Evaluates models using performance metrics
   - Selects the best-performing model
   - Saves trained model as model.pkl


## Prediction Pipeline

Implemented in:
src/pipeline/predict_pipeline.py

Prediction flow:
1. User enters data through Flask UI
2. CustomData class converts inputs into a pandas DataFrame
3. PredictPipeline loads preprocessor and trained model
4. Input data is transformed
5. Maths score is predicted using the trained model


## CustomData Class

The CustomData class converts raw user inputs into a structured pandas DataFrame that matches the training schema.

Input features:
- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score


## Flask Application Flow

1. User opens the home page
2. Enters student details
3. Submits the form
4. Flask route:
   - Creates CustomData object
   - Calls prediction pipeline
   - Displays predicted Maths score on the UI


## How to Run the Project Locally

### 1. Clone the repository
```
git clone https://github.com/Pillapavan/ML_Project.git
cd ML_Project
```

### 2. Create and activate virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run Flask app
```
python app.py
```

### 5. Open browser
```
http://127.0.0.1:5000/
```

## Author

Pavan Pilla  
GitHub: https://github.com/Pillapavan
