# Performance Analysis of Text Processing for Laptop Reviews

The goal of this project is to analyze the effectiveness of Natural Language Processing (NLP) techniques and Machine Learning (ML) models in performing sentiment analysis on Amazon laptop reviews. The study evaluates the impact of NLP pre-processing techniques, such as Porter stemming, lemmatization, and stopword removal, on sentiment classification. Multiple ML models, including Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Transformer, and TabMLP, are implemented to compare their performance under various conditions.

## 🚀 Getting Started

These instructions will help you set up and run the project on your local machine.

- **Table of Contents**

- **Project Description**
- **Features**
- **Installation**
- **Usage**
- **Implementation**
- **Evaluation Metrics**

## Project Description

This research investigates sentiment analysis using NLP techniques and ML models to classify Amazon laptop reviews into positive and negative sentiments. A dataset containing 414 reviews was manually curated and processed. The ML models were evaluated both with and without NLP pre-processing techniques. Performance was assessed using metrics such as accuracy, precision, recall, and F1-score. The study's goal is to identify the optimal combination of preprocessing techniques and models for sentiment analysis in this context.

## Features

1. #### NLP Preprocessing Techniques:
- Stopword Removal
- Porter Stemming
- Lemmatization

2. #### Tree-based ML Models:
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

3. #### Advanced Models:
- Transformers
- TabMLP

4. #### Comprehensive Dataset:
- 414 Amazon laptop reviews
- Preprocessed to ensure data quality and balance

5. #### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Installation

#### Prerequisites
- Python
- Jupyter Notebook

### Setup
#### Clone the repository:

```
git clone https://github.com/<your-repo>/text-processing-for-laptop-reviews.git
```

#### Navigate to the project directory:
```
cd text-processing-for-laptop-reviews
```

#### Install dependencies:
```
pip install -r requirements.txt
```
### Usage
#### Prepare Dataset: 

#### Place your dataset in the same directory as your main code file
Here is a simple three-step process to run an ```.ipynb``` file:

### Step 1: Install Jupyter Notebook
If you haven’t already, install Jupyter Notebook. Open your command prompt (or terminal) and type:
```
pip install notebook
```
### Step 2: Launch Jupyter Notebook
Navigate to the directory containing your .ipynb file using the command line. Then, launch Jupyter Notebook by typing:
```
jupyter notebook
```
This command will open Jupyter in your default web browser.

### Step 3: Open and Run the Notebook
In the Jupyter interface, browse to the folder containing your .ipynb file and click on the file to open it.
To run the code in a cell, click the "Run" button or press Shift + Enter. Repeat this for each cell, or go to Kernel > Restart & Run All to run all cells in sequence.
You can now interact with and run your Jupyter Notebook file!

## Implementation

### Data Preprocessing
The dataset of 414 Amazon laptop reviews is cleaned and preprocessed to ensure quality and suitability for sentiment analysis. Key steps include:
1. **Remove Duplicates and Null Values**:
   - Any duplicate entries or null values in the dataset are removed to prevent skewed results or inaccurate model training.
   - After this step, the dataset is reduced to 399 unique rows.
2. **Apply NLP Techniques**:
   - **Stopword Removal**: Commonly used words like "and," "the," or "is" that do not add meaningful context to sentiment analysis are eliminated.
   - **Porter Stemming**: Reduces words to their root form by removing suffixes (e.g., "running" becomes "run"). This helps minimize feature space while retaining meaning.
   - **Lemmatization**: Converts words to their base or dictionary form (e.g., "better" becomes "good"). This technique ensures grammatical correctness and preserves semantic meaning.

### Model Training
1. **Data Splitting**:
   - The dataset is split into three parts:  
     - **Training Set**: 80% of the data, used to train models.  
     - **Validation Set**: 10% of the data, used to tune model hyperparameters.  
     - **Testing Set**: 10% of the data, used for evaluating final model performance.  
2. **Model Implementation**:
   - A variety of machine learning models are trained, including:  
     - Tree-based models (Decision Tree, Random Forest, AdaBoost, Gradient Boosting).  
     - Advanced models (Transformers, TabMLP).  
   - Models are evaluated with and without NLP preprocessing to understand the impact of these techniques on performance.

### Evaluation
1. **Comparison**:
   - Models are assessed based on their performance with and without each NLP preprocessing technique.
   - Results are analyzed to determine the optimal combination of model and preprocessing method.
2. **Visualization**:
   - Confusion matrices and performance metrics are generated for each model and preprocessing approach, providing insights into accuracy, precision, recall, and F1-score.

---

## Evaluation Metrics

1. **Accuracy**: Measures the percentage of correct predictions out of all predictions made.  
   Formula:  
   \[
   Accuracy = \frac{(True\ Positives + True\ Negatives)}{Total\ Predictions}
   \]
2. **Precision**: Evaluates the relevance of the positive predictions by calculating the ratio of true positives to all predicted positives.  
   Formula:  
   \[
   Precision = \frac{True\ Positives}{(True\ Positives + False\ Positives)}
   \]
3. **Recall**: Assesses the completeness by finding the ratio of true positives to all actual positives.  
   Formula:  
   \[
   Recall = \frac{True\ Positives}{(True\ Positives + False\ Negatives)}
   \]
4. **F1-Score**: The harmonic mean of precision and recall, offering a balanced measure of the two.  
   Formula:  
   \[
   F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
   \]

---

## Findings

1. **Decision Trees**:
   - Show the best performance when **Stopword Removal** is applied. This indicates that removing irrelevant words helps reduce noise for this model.
2. **Random Forest**:
   - Achieves high accuracy without any NLP preprocessing, demonstrating its robustness to raw data.
3. **AdaBoost**:
   - Benefits significantly from **Lemmatization**, suggesting that grammatical correctness improves its ability to discern patterns in the data.
4. **Gradient Boosting**:
   - Works optimally with **Porter Stemming**, where simplifying words to their roots enhances performance.
5. **Transformers**:
   - Show the most improvement with **Lemmatization**, which helps them better understand the context of the reviews.

### Conclusion
This study reveals how NLP preprocessing techniques influence model performance, providing actionable insights for selecting the right combination of preprocessing and model. The findings highlight the potential to enhance text analysis workflows, ultimately benefiting product quality and customer satisfaction.
