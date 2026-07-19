import joblib 
from pathlib import Path

MODEL_PATH = Path(r'Spam_detection 2_0\models\model.joblib')


def predict_msg(msg: str) -> str :
    """
    Predicts whether a given message is spam or not using a pre-trained model.

    Args:
        msg (str): The message to be classified.

    Returns:
        tuple: A tuple containing the predicted label ('spam' or 'ham') and the confidence score.
    """
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    prediction = model.predict([msg])[0]

    return prediction

if __name__ == "__main__":

    sample_text = "Congratulations! You have won a free prize. Call now!"
    label  = predict_msg(sample_text)
    print(f"Prediction {label}")
