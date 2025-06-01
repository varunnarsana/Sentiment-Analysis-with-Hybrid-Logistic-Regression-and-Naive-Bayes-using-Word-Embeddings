# Import required libraries
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
import re
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Download and extract the IMDB dataset
!wget -q -nc "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
!tar -xf aclImdb_v1.tar.gz

def load_imdb_data(base_path='/content/aclImdb', max_samples=5000):
    """
    Load IMDb data from the extracted files
    base_path: Path to the extracted IMDb dataset
    max_samples: Maximum number of samples to load
    """
    texts = []
    labels = []
    
    # Load positive reviews from training set
    pos_path = os.path.join(base_path, 'train', 'pos')
    print(f"Loading positive reviews from: {pos_path}")
    
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Directory not found: {pos_path}")
        
    pos_files = sorted([f for f in os.listdir(pos_path) if f.endswith('.txt')])
    pos_files = pos_files[:max_samples//2]
    
    for filename in tqdm(pos_files, desc="Loading positive reviews"):
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(1)
    
    # Load negative reviews from training set
    neg_path = os.path.join(base_path, 'train', 'neg')
    print(f"Loading negative reviews from: {neg_path}")
    
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"Directory not found: {neg_path}")
        
    neg_files = sorted([f for f in os.listdir(neg_path) if f.endswith('.txt')])
    neg_files = neg_files[:max_samples//2]
    
    for filename in tqdm(neg_files, desc="Loading negative reviews"):
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(0)
    
    print(f"Loaded {len(texts)} reviews in total")
    return texts, labels

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.word2vec_model = None
        self.lr_model = None
        self.nb_model = None
        
    def preprocess_text(self, text):
        """
        Preprocess the input text
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return tokens
    
    def load_word2vec(self):
        """Load GloVe vectors"""
        print("Downloading GloVe vectors...")
        
        # Download GloVe vectors if they don't exist
        if not os.path.exists('glove.6B.100d.txt'):
            !wget -q -nc "https://nlp.stanford.edu/data/glove.6B.zip"
            !unzip -q glove.6B.zip
        
        # Convert GloVe to Word2Vec format
        from gensim.scripts.glove2word2vec import glove2word2vec
        
        if not os.path.exists('glove_word2vec.txt'):
            glove2word2vec("glove.6B.100d.txt", "glove_word2vec.txt")
        
        print("Loading word vectors...")
        self.word2vec_model = KeyedVectors.load_word2vec_format("glove_word2vec.txt")
        print("Word vectors loaded successfully!")
    
    def get_document_vector(self, tokens):
        """Convert document tokens to average word vector"""
        vectors = []
        for token in tokens:
            if token in self.word2vec_model:
                vectors.append(self.word2vec_model[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)
    
    def prepare_data(self, texts, labels):
        """Prepare data by converting texts to document vectors"""
        X = np.array([self.get_document_vector(self.preprocess_text(text)) 
                     for text in tqdm(texts, desc="Preparing data")])
        y = np.array(labels)
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train both Logistic Regression and Naive Bayes models"""
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(max_iter=1000)
        self.lr_model.fit(X_train, y_train)
        
        print("Training Naive Bayes...")
        self.nb_model = MultinomialNB()
        X_train_nb = X_train - X_train.min()
        self.nb_model.fit(X_train_nb, y_train)
    
    def predict_hybrid(self, X):
        """Combine predictions from both models"""
        X_nb = X - X.min()
        lr_prob = self.lr_model.predict_proba(X)
        nb_prob = self.nb_model.predict_proba(X_nb)
        combined_prob = (lr_prob + nb_prob) / 2
        return np.argmax(combined_prob, axis=1)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate individual models and hybrid model"""
        X_test_nb = X_test - X_test.min()
        
        lr_pred = self.lr_model.predict(X_test)
        nb_pred = self.nb_model.predict(X_test_nb)
        hybrid_pred = self.predict_hybrid(X_test)
        
        print("\nLogistic Regression Results:")
        print(classification_report(y_test, lr_pred))
        
        print("\nNaive Bayes Results:")
        print(classification_report(y_test, nb_pred))
        
        print("\nHybrid Model Results:")
        print(classification_report(y_test, hybrid_pred))

def main():
    try:
        # Check if the dataset directory exists
        if not os.path.exists('/content/aclImdb'):
            print("Dataset not found. Downloading and extracting...")
            !wget -q -nc "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            !tar -xf aclImdb_v1.tar.gz
        
        # Print the current directory structure
        print("\nCurrent directory structure:")
        !ls -l /content/aclImdb/train
        
        # Load the IMDb dataset
        print("\nLoading IMDb dataset...")
        texts, labels = load_imdb_data(max_samples=5000)  # Using 5000 samples
        
        if len(texts) == 0:
            raise ValueError("No texts were loaded. Please check the dataset path.")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Initialize the analyzer
        analyzer = SentimentAnalyzer()
        
        # Load word vectors
        analyzer.load_word2vec()
        
        # Prepare data
        print("\nPreparing training data...")
        X_train_vec, y_train = analyzer.prepare_data(X_train, y_train)
        print("\nPreparing test data...")
        X_test_vec, y_test = analyzer.prepare_data(X_test, y_test)
        
        # Train models
        analyzer.train_models(X_train_vec, y_train)
        
        # Evaluate models
        analyzer.evaluate_models(X_test_vec, y_test)
        
        # Test with sample reviews
        sample_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible waste of time. The plot made no sense at all.",
            "It was okay, not great but not terrible either."
        ]
        
        X_samples = np.array([analyzer.get_document_vector(analyzer.preprocess_text(text)) 
                             for text in sample_texts])
        predictions = analyzer.predict_hybrid(X_samples)
        
        print("\nSample Predictions:")
        for text, pred in zip(sample_texts, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {text}\nPredicted sentiment: {sentiment}\n")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nDebug information:")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
        if os.path.exists('/content/aclImdb'):
            print("IMDb directory contents:", os.listdir('/content/aclImdb'))

if __name__ == "__main__":
    main()
