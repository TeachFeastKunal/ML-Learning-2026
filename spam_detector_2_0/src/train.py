import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from utils import load_data , standardize_columns , feature_engineering , clean_msg

MODEL_PATH = Path(r'Spam_detection 2_0\models\model.joblib')
DATA_PATH = Path(r'Spam_detection 2_0\data\spam.csv')

def main():
    df = load_data(DATA_PATH)
    df = df.pipe(standardize_columns).pipe(feature_engineering).pipe(clean_msg)

    X = df['msg']
    Y = df['label']

    x_train  , x_test , y_train , y_test = train_test_split(X , Y , stratify = Y , test_size = 0.2 , random_state = 42)

    pipeline = Pipeline([
        ('tfidf' , TfidfVectorizer(max_features = 3000 , ngram_range = (1 , 2))),
        ('svc' , LinearSVC(C = 1.0 , random_state= 42 , dual = False))
    ])

    pipeline.fit(x_train , y_train)
    Path('models').mkdir(parents = True , exist_ok = True)

    joblib.dump({
        "model" : pipeline , 
        "x_test" : x_test , 
        "y_test" : y_test 
    }, MODEL_PATH)

    print(f"Model Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
