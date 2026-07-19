# Project C - Spam Detector 2.0

A machine learning spam detection project that classifies SMS messages as **spam** or **ham** using a TF-IDF + LinearSVC pipeline. The project includes exploratory analysis in a notebook and a cleaner script-based workflow for training, evaluation, and prediction.[1][2][3][4]

## Project structure

```text
spam_detector_2_0/
├── eda_and_modeling.ipynb
├── train.py
├── evaluate.py
├── predict.py
├── utils.py
├── spam.csv
├── requirements.txt
└── README.md
```

The notebook contains EDA, preprocessing experiments, and modeling work, while the Python scripts provide a more reusable workflow closer to standard project structure.[1][2][3]

## Features

- Cleans and standardizes the raw SMS dataset by renaming columns, dropping extra unnamed columns, and removing duplicates.[5][1]
- Adds simple text-based features such as character count, word count, and sentence count during preprocessing.[5]
- Uses NLTK-based tokenization, stopword removal, and lemmatization or stemming for text cleaning.[5][1]
- Trains a TF-IDF plus LinearSVC pipeline for SMS spam classification.[2]
- Saves the trained model and test split with Joblib for later evaluation and prediction.[2]
- Evaluates the model with accuracy, precision, recall, F1-score, confusion matrix, and classification report.[3]

## Tech stack

- Python
- pandas, numpy
- scikit-learn
- NLTK
- Joblib
- matplotlib, seaborn
- Jupyter Notebook
- kagglehub (used in the notebook workflow)[1]

## Dataset

The project uses the SMS Spam Collection dataset stored in `spam.csv`, and the notebook also shows a KaggleHub-based download workflow for the same dataset source.[6][1]

After preprocessing shown in the notebook, the working dataset has `label` and `msg` as the main columns used for training, with duplicate rows removed and extra unnamed columns dropped.[1][5]

## Setup

1. Clone or download the project files.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources if needed. The project scripts already call `nltk.download(...)` inside preprocessing, so they should fetch required resources on first run.[5]

## How to run

### 1. Train the model

```bash
python train.py
```

This script loads `spam.csv`, standardizes columns, engineers simple features, cleans the message text, splits the data, trains a `TfidfVectorizer` + `LinearSVC` pipeline, and saves the result as a Joblib file.[2][5]

### 2. Evaluate the model

```bash
python evaluate.py
```

This script loads the saved artifact, runs prediction on the held-out test set, and prints classification metrics along with a confusion matrix and classification report.[3]

### 3. Predict on a sample message

```bash
python predict.py
```

This script loads the saved model artifact and predicts whether a message is spam or ham.[4]

## Model details

The training script uses:

- `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`
- `LinearSVC(C=1.0, random_state=42, dual=False)`[2]

This gives a strong baseline for short text classification problems like SMS spam detection.[2]

## Preprocessing pipeline

The reusable preprocessing logic in `utils.py` includes:

- `load_data()` for reading the CSV with `latin-1` encoding.[5]
- `standardize_columns()` for renaming `v1` to `label` and `v2` to `msg`, removing extra columns, and dropping duplicates.[5]
- `feature_engineering()` for character, word, and sentence counts.[5]
- `clean_msg()` for lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization or stemming.[5]

This structure keeps the code modular without overcomplicating the project.[5]

## Notebook workflow

The notebook is useful for:

- understanding the raw dataset,
- checking class imbalance,
- exploring message length patterns,
- testing preprocessing ideas,
- comparing modeling approaches before finalizing the script version.[1]

The notebook notes that the dataset is class-imbalanced and that spam messages tend to be longer than ham messages, which supports evaluating the classifier with precision, recall, and F1-score instead of relying only on accuracy.[1]

## Notes

- `predict.py` expects the saved model artifact path to exist before prediction is run.[4]
- The current scripts reference a specific saved model filename/path, so keeping file paths consistent is important for smooth execution.[2][3][4]
- The project is a good example of moving from notebook experimentation to a small script-based ML workflow.[1][2]

## Future improvements

- Add argument parsing with `argparse` for flexible input paths and prediction messages.
- Save metrics to a file instead of only printing them.
- Add a small FastAPI or Flask app for deployment.
- Add unit tests for preprocessing functions.
- Track experiments with MLflow or DVC in a later version.

## Author note

This project is a strong beginner-to-intermediate ML portfolio piece because it shows both notebook-based experimentation and script-based implementation in a simple, understandable structure.[1][2][5]
