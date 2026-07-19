import joblib 
from pathlib import Path
from sklearn.metrics import precision_score , recall_score , accuracy_score , f1_score , classification_report , confusion_matrix

MODEL_PATH = Path(r'Spam_detection 2_0\models\model.joblib')

def main():
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    x_test = model_data['x_test']
    y_test = model_data['y_test']

    y_pred = model.predict(x_test)


    svc_accuracy = accuracy_score(y_test, y_pred)
    svc_precision = precision_score(y_test, y_pred, pos_label='spam')
    svc_recall = recall_score(y_test, y_pred, pos_label='spam')
    svc_f1 = f1_score(y_test, y_pred, pos_label='spam')
    svc_conf_matrix = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

    print("================ LINEARSVC METRICS ================")
    print(f"Accuracy  : {svc_accuracy:.4f}")
    print(f"Precision : {svc_precision:.4f}  <-- (Out of all predicted spam, how many were actually spam)")
    print(f"Recall    : {svc_recall:.4f}     <-- (Out of all real spam, how many did we catch)")
    print(f"F1-Score  : {svc_f1:.4f}         <-- (Harmonic balance of Precision and Recall)")
    print("================================================================")
    print("\nConfusion Matrix:")
    print(f"                 Predicted Ham   Predicted Spam")
    print(f"Actual Ham  :        {svc_conf_matrix[0][0]:<15} {svc_conf_matrix[0][1]}")
    print(f"Actual Spam :        {svc_conf_matrix[1][0]:<15} {svc_conf_matrix[1][1]}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()