# Classification-and-Summarization-of-Medical-Abstracts
### Multi-Label Classification and Extractive Summarization of Medical Abstracts.

This project was developed in collaboration with Paolo Caggiano for the Text Mining course in the Master Degree in Data Science.<br><br>
The growing volume of biomedical literature presents a significant challenge in quickly retrieving relevant information. Our project addresses this by combining traditional Natural Language Processing techniques to classify and summarize medical abstracts.<br><br>
The first part of the project focuses on <strong>Multi-Class Multi-Label classification</strong> of medical documents across five diagnostic categories. We explore and compare multiple combinations of text <strong>preprocessing</strong> (basic cleaning, stop-word removal, lemmatization), <strong>feature extraction</strong> (BoW, TF, TF-IDF, word embeddings), and <strong>feature selection</strong> methods (rare word removal, PCA). We then apply <strong>four different classifiers</strong>: Naive Bayes, Decision Trees, Random Forests, and SVMs, evaluating performance through <strong>F-score with 5-fold cross-validation</strong>. Best results (71.1% F1 score) were obtained using pretrained biomedical word embeddings and SVMs.<br><br>
In the second part, we apply <strong>extractive summarization techniques</strong> (Graph-based PageRank and Latent Semantic Analysis) to the abstracts. Summaries were evaluated using <strong>ROUGE scores</strong> against article titles and by assessing their utility in reproducing classification results. Our graph-based summarizer outperformed both LSA and random baselines, retaining more relevant information for downstream tasks.<br><br>
The project showcases our ability to apply a full NLP pipeline, from document representation to classification and summarization, emphasizing model evaluation and empirical comparison of classical techniques. Future work is directed toward integrating contextualized embeddings like Med-BERT and exploring abstractive summarization.<br><br>
The complete code and methodology are detailed in the Github page and in the downloadable report below.

### Dataset
To perform our evaluation, we utilized the dataset coming from [“Evaluating unsupervised text classification: zero-shot and similaritybased approaches](https://dl.acm.org/doi/abs/10.1145/3582768.3582795), available on [Github](https://github.com/sebischair/medical-abstracts-tc-corpus).

### Notebooks
There are 6 notebooks in this project, here they are listed in chronological order:

1. **PreProcessing.ipynb**
Here we take the original dataset and apply some modifications to its structure in order to make it appropriate to the task of MCML classification.
This dataset is stored in "Data/No PreProcessing".
We then apply Stopwords Removal and Lemmatization, and save the resulting data respectively in "Data/StopWords" and "Data/StopWords + Lemmatization".

2. **Feature Extraction with Cross Validation.ipynb**
In this notebook we apply MLMC classification with a 5-fold cross validation over the 6 combinations of PreProcessing and Feature Extraction.

3. **Feature Selection.ipynb**
In this notebook we apply two types of Feature Selection: the first one is an easy removal of the words that occur just once in the whole text, the second is Principal Component Analysis.

4. **W2V.ipynb**
In this notebook we train a Word2Vec model over the data, train the classifier using the created word embeddings, and compute the evaluation metrics.
After seeing that StopWords Removal + Lemmatization was the best combination, we tried also changing the model parameters (vector size and window).

5. **Pretrained W2v.ipynb**
In this notebook we applied a pretrained W2V. For reason of space, we do not include in this folder the pretrained model, but it can be downloaded from the original website: https://bio.nlplab.org/.

6. **Text Summarization.ipynb**
In this notebook we apply text summarization with a graph-based model and LSA. We then evaluate the performances with rouge metrics and classification.
