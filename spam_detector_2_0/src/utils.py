import pandas as pd
import re , string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer , PorterStemmer 


def load_data(path : str ) -> pd.DataFrame : 
    df = pd.read_csv(path , encoding = 'latin-1')
    return df

def standardize_columns(df : pd.DataFrame) -> pd.DataFrame : 
    df = df.copy()
    if 'v1' and 'v2' in df.columns :
        df = df.rename(columns = {'v1' : 'label' , 'v2' : 'msg'})
    extra_columns = set(df.columns) - {'label' , 'msg'}

    if extra_columns:
        df = df.drop(extra_columns , axis = 1)
    
    if df.duplicated().sum() > 0 : 
        df = df.drop_duplicates()

    return df

def feature_engineering(df : pd.DataFrame) -> pd.DataFrame : 
    df = df.copy()

    df['char_count'] = df['msg'].apply(len)
    df['word_count'] = df['msg'].apply(lambda x : len(x.split()))
    df['sentence_count'] = df['msg'].apply(lambda x : len([line for line in re.split(r'[.!?]+' , x , flags = re.U) if line.split()]))

    return df

def clean_msg(df : pd.DataFrame , strategy = 'lemmatize') -> pd.DataFrame : 
    df = df.copy()

    nltk.download('punkt_tab' , quiet = True)
    nltk.download('stopwords' , quiet = True)
    nltk.download('wordnet' , quiet = True)
    nltk.download('omw-1.4' , quiet = True)

    STOPWORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    STEMMER = PorterStemmer()

    def tokenize_text(text : str) -> str : 
        if not isinstance(text , str):
            return ''
        
        text = text.lower()
        text = text.translate(str.maketrans('' , '' , string.punctuation))

        tokens = word_tokenize(text)

        cleaned_tokens = [word for word in tokens if word not in STOPWORDS and word.strip()]

        if strategy == 'lemmatize':
            final_tokens = [LEMMATIZER.lemmatize(word) for word in cleaned_tokens]
        elif strategy == 'stem' :
            final_tokens = [STEMMER.stem(word) for word in cleaned_tokens]
        
        return ' '.join(final_tokens)
    
    df['cleaned_msg'] = df['msg'].apply(lambda x : tokenize_text(x))

    return df