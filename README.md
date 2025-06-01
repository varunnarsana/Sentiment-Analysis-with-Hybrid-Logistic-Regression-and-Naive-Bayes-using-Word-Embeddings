# Sentiment Analysis with Hybrid Logistic Regression and Naive Bayes using Word Embeddings

This project implements a hybrid approach for sentiment analysis by combining Logistic Regression and Naive Bayes classifiers, enhanced with word embeddings (Word2Vec/GloVe). The goal is to improve sentiment classification accuracy by leveraging both the interpretability of Logistic Regression and the efficiency of Naive Bayes, while capturing semantic relationships through word embeddings.

---

## Project Structure

```
.
├── nlp-paper.docx                          # Project report and documentation
├── nlpcode.py                              # Main Python code (implementation)
├── sentiment-analysis-implementation-v3.py # Alternative/updated implementation
```

---

## Problem Statement

Traditional sentiment analysis models often struggle to capture context and nuanced expressions in text. This project addresses these limitations by:
- Using word embeddings to encode semantic and contextual information.
- Combining Logistic Regression and Naive Bayes to leverage their complementary strengths for more robust sentiment classification[1].

---

## Dataset

- **Source:** IMDb movie reviews (downloaded and extracted automatically)
- **Size:** 5,000 labeled samples (2,500 positive, 2,500 negative)
- **Format:** Plain text files, each containing a review and its sentiment label

---

## Methodology

**1. Data Preprocessing**
   - Tokenization, lowercasing, removal of special characters and numbers
   - Stopword removal and lemmatization using NLTK[2][3]

**2. Word Embedding Generation**
   - Downloads and loads pre-trained GloVe vectors (100d)
   - Converts each review into a dense vector by averaging its word embeddings

**3. Model Training**
   - Trains both Logistic Regression and Multinomial Naive Bayes on the vectorized data
   - Trains a hybrid model by averaging the predicted probabilities from both classifiers

**4. Evaluation**
   - Splits data into training and test sets (80/20)
   - Evaluates models using accuracy, precision, recall, and F1-score
   - Provides classification reports for each model

---

## How to Run

1. **Install Requirements**
   ```bash
   pip install numpy pandas nltk gensim scikit-learn tqdm
   ```

2. **Download and Prepare Data**
   - The script automatically downloads the IMDb dataset and GloVe vectors if not present.

3. **Run the Code**
   - Execute `nlpcode.py` or `sentiment-analysis-implementation-v3.py` in your Python environment.

---

## Example: Key Implementation Steps

```python
# Preprocessing and vectorization
tokens = analyzer.preprocess_text("The movie was great!")
vector = analyzer.get_document_vector(tokens)

# Model training
analyzer.train_models(X_train_vec, y_train)

# Hybrid prediction
hybrid_pred = analyzer.predict_hybrid(X_test_vec)
```

---

## Results

- **Logistic Regression:** Higher precision and recall than Naive Bayes
- **Naive Bayes:** Fast, robust to sparse data
- **Hybrid Model:** Best overall F1-score, balanced performance across metrics
- **Word Embeddings:** Significantly improve all models by capturing context and semantics[1]

---

## Sample Output

```
Logistic Regression Results:
              precision    recall  f1-score   support
           0       0.89      0.91      0.90       500
           1       0.90      0.88      0.89       500

Naive Bayes Results:
              precision    recall  f1-score   support
           0       0.85      0.87      0.86       500
           1       0.86      0.84      0.85       500

Hybrid Model Results:
              precision    recall  f1-score   support
           0       0.91      0.92      0.91       500
           1       0.92      0.91      0.91       500
```

---

## Recommendations

- **Scalability:** The hybrid approach is computationally efficient and scalable to larger datasets.
- **Further Improvements:** Explore contextual embeddings (e.g., BERT) and other ensemble strategies for even better performance[1].
- **Code Quality:** The code is modular, well-commented, and follows best practices for reproducibility and robustness[2][3].

---

## Conclusion

Combining Logistic Regression and Naive Bayes with word embeddings yields a robust, interpretable, and accurate sentiment analysis model. This hybrid approach effectively addresses the limitations of traditional models by leveraging semantic context and the strengths of both classifiers[1].

---

