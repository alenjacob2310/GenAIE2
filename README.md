# Gen AI E2 Hands-on Lab - Comprehensive Training Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Lab 1: NLP Disaster Tweets Prediction with TF-IDF](#lab-1-nlp-disaster-tweets-prediction-with-tf-idf)
4. [Lab 2: Knowledge Graph Construction and RAG](#lab-2-knowledge-graph-construction-and-rag)
5. [Installation Guide](#installation-guide)
6. [Troubleshooting](#troubleshooting)
7. [Learning Outcomes](#learning-outcomes)
8. [Additional Resources](#additional-resources)

---

## 🎯 Project Overview

This repository contains two comprehensive hands-on labs designed for training in **Natural Language Processing (NLP)** and **Knowledge Graph** technologies:

### **Lab 1: NLP Disaster Tweets Prediction**
A complete machine learning pipeline for classifying disaster-related tweets using TF-IDF vectorization and Random Forest classification.

### **Lab 2: Knowledge Graph Construction & RAG**
An end-to-end implementation of Knowledge Graph creation from DBpedia data and Retrieval-Augmented Generation (RAG) using GPT-2.

---

## 📚 Prerequisites

### Required Knowledge
- **Python Programming**: Intermediate level
- **Basic Machine Learning**: Understanding of classification, training/testing splits
- **NLP Fundamentals**: Text preprocessing, tokenization, stopwords
- **Data Structures**: DataFrames, arrays, dictionaries
- **Graph Theory**: Basic concepts of nodes, edges, and relationships (for Lab 2)

### Software Requirements
```
Python >= 3.7
Jupyter Notebook or JupyterLab
```

### Required Libraries
```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
nltk >= 3.6
networkx >= 2.6
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0
transformers >= 4.10.0
```

---

## 🚀 Lab 1: NLP Disaster Tweets Prediction with TF-IDF

### 📖 Introduction

This lab teaches you how to build a complete NLP pipeline to classify tweets as disaster-related or not. You'll learn data preprocessing, text cleaning, feature extraction using TF-IDF, and model training.

### 🎯 Learning Objectives

By completing this lab, you will:
1. Load and explore textual datasets
2. Handle missing values in real-world data
3. Clean and preprocess text data (URLs, HTML, emojis, punctuation)
4. Apply advanced NLP techniques (stopword removal, lemmatization)
5. Create document-term matrices using Count Vectorizer
6. Implement TF-IDF vectorization
7. Train and evaluate a Random Forest classifier
8. Interpret classification metrics and confusion matrices

---

### 📂 Dataset Description

**Source**: Natural Language Processing with Disaster Tweets

Each tweet contains:
- **id**: Unique identifier
- **text**: The tweet content
- **keyword**: A keyword from the tweet (may be blank)
- **location**: Location where tweet was sent (may be blank)
- **target**: Binary label (1 = real disaster, 0 = not a disaster)

**Expected Input**: `train.csv` file in the working directory

---

### 🔧 Task 1: Data Loading and Handling Missing Values

#### Objective
Load the dataset, explore its structure, and handle missing values appropriately.

#### Instructions
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Explore the data
print(df.head())
print(df.info())
print(df.isnull().sum())

# Drop columns with too many missing values
df.drop(['keyword', 'location'], inplace=True, axis=1)

# Create a cleaned version
data_cleaned = df.copy()
```

#### 📊 Expected Output
- DataFrame with 3 columns: `id`, `text`, `target`
- No missing values in critical columns

#### 💡 Key Concepts
- **Missing Data**: Columns with >50% missing values should typically be dropped
- **Data Types**: Ensure text columns are strings and target is numeric
- **Data Exploration**: Use `.info()`, `.describe()`, `.head()` to understand your data

---

### 🧹 Task 2: Data Cleaning and Text Preprocessing

#### Objective
Transform raw tweet text into clean, normalized text suitable for machine learning.

#### Why Text Preprocessing?
Raw text contains noise that can confuse ML models:
- URLs don't carry sentiment information
- HTML tags are artifacts
- Punctuation adds unnecessary features
- Stopwords (the, is, at) add noise
- Different word forms (running, runs, ran) should be unified

#### Instructions

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Define comprehensive cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove HTML tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Remove stopwords
    text = [word for word in text.split() if word not in sw]
    
    # Lemmatization (reduce words to base form)
    text = [lemmatizer.lemmatize(word) for word in text]
    
    # Join back into string
    text = " ".join(text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)
clean_df = df.head()
```

#### 📊 Example Transformation

**Before**:
```
"Forest fire near La Ronge Sask. Canada http://t.co/abcd1234 🔥"
```

**After**:
```
"forest fire near la ronge sask canada"
```

#### 💡 Key Concepts
- **Lemmatization vs Stemming**: Lemmatization produces actual words (better → good), while stemming may not (running → run)
- **Stopwords**: Common words that don't carry much meaning
- **Regular Expressions**: Powerful pattern matching for text cleaning
- **Unicode Handling**: Proper removal of emojis and special characters

---

### 📊 Task 3: Create a Document-Term Matrix using Count Vectorizer

#### Objective
Transform text into numerical features that machine learning algorithms can process.

#### What is a Document-Term Matrix?
A matrix where:
- **Rows** = Documents (tweets)
- **Columns** = Unique words (features)
- **Values** = Count of each word in each document

#### Instructions

```python
from sklearn.feature_extraction.text import CountVectorizer

# Use first two rows for demonstration
sample_corpora = df['text'].iloc[:2].values

# Initialize Count Vectorizer
count_vectorizer = CountVectorizer()

# Fit and transform
wm = count_vectorizer.fit_transform(sample_corpora)

# Create document names
doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]

# Extract feature names (tokens)
feat_names = count_vectorizer.get_feature_names_out()

# Create DataFrame
sample_df = pd.DataFrame(
    data=wm.toarray(), 
    index=doc_names, 
    columns=feat_names
)

print(sample_df)
```

#### 📊 Example Output

|      | fire | forest | near | canada | earthquake | building |
|------|------|--------|------|--------|------------|----------|
| Doc0 | 1    | 1      | 1    | 1      | 0          | 0        |
| Doc1 | 0    | 0      | 0    | 0      | 1          | 1        |

#### 💡 Key Concepts
- **Vocabulary**: The set of all unique words across all documents
- **Sparse Matrix**: Most values are 0 (words don't appear in most documents)
- **Feature Extraction**: Converting text to numerical format
- **Dimensionality**: Number of columns = vocabulary size

---

### 🎯 Task 4: Apply TF-IDF Vectorization

#### Objective
Create weighted feature vectors where word importance is based on TF-IDF scores.

#### Why TF-IDF instead of Count Vectorizer?

**Count Vectorizer** counts word occurrences but treats all words equally.

**TF-IDF** (Term Frequency - Inverse Document Frequency):
- **TF**: How often a word appears in a document
- **IDF**: How rare a word is across all documents
- **Result**: Common words get lower scores, rare but meaningful words get higher scores

**Formula**:
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)
IDF(word) = log(Total Documents / Documents containing word)
```

#### Instructions

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values,
    df['target'].values,
    test_size=0.2,        # 80% train, 20% test
    random_state=123,      # Reproducibility
    stratify=df['target'].values  # Maintain class distribution
)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit on training data and transform
tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

# Transform test data (use fitted vectorizer)
tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

print("Training Vector Shape:", tfidf_train_vectors.shape)
print("Test Vector Shape:", tfidf_test_vectors.shape)
```

#### 📊 Expected Output
```
Training Vector Shape: (6090, ~12000)
Test Vector Shape: (1523, ~12000)
```

#### 💡 Key Concepts
- **Train-Test Split**: Essential to evaluate model on unseen data
- **Stratification**: Ensures both sets have similar class distribution
- **Fit vs Transform**: 
  - `fit_transform()` on training data (learns vocabulary)
  - `transform()` on test data (uses learned vocabulary)
- **Feature Space**: Typically 11,000-14,000 features for this dataset

#### ⚠️ Common Pitfalls
- Never fit on test data (causes data leakage)
- Always use the same vectorizer for train and test
- Ensure test set doesn't introduce new vocabulary

---

### 🤖 Task 5: Train and Evaluate a Random Forest Classifier

#### Objective
Build a classification model and evaluate its performance using multiple metrics.

#### Why Random Forest?
- **Ensemble Method**: Combines multiple decision trees
- **Handles High Dimensions**: Works well with many features
- **Robust**: Less prone to overfitting than single decision trees
- **Feature Importance**: Can identify important words

#### Instructions

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Initialize Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)

# Train the model
classifier.fit(tfidf_train_vectors, y_train)

# Make predictions
y_pred = classifier.predict(tfidf_test_vectors)

# Calculate F1 Score
f1score = round(f1_score(y_test, y_pred), 2)

# Generate Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix
group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
group_counts = [f"{value}" for value in cnf_matrix.flatten()]
labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues', cbar=True)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
```

#### 📊 Understanding the Metrics

##### Confusion Matrix

|                    | Predicted: No Disaster | Predicted: Disaster |
|--------------------|------------------------|---------------------|
| **Actual: No Disaster** | True Negative (TN)     | False Positive (FP) |
| **Actual: Disaster**    | False Negative (FN)    | True Positive (TP)  |

##### Classification Metrics

**Precision**: Of all predicted disasters, how many were correct?
```
Precision = TP / (TP + FP)
```

**Recall**: Of all actual disasters, how many did we catch?
```
Recall = TP / (TP + FN)
```

**F1-Score**: Harmonic mean of Precision and Recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / Total
```

#### 💡 Expected Results
- **F1-Score**: ~0.70-0.78 (depending on preprocessing quality)
- **Accuracy**: ~0.75-0.82
- Better recall for non-disaster tweets typically

#### 🔍 Model Interpretation

**High Precision, Low Recall**: Model is conservative (few false alarms but misses real disasters)

**High Recall, Low Precision**: Model is aggressive (catches most disasters but many false alarms)

**Balanced F1-Score**: Good trade-off for disaster prediction

---

## 🕸️ Lab 2: Knowledge Graph Construction and RAG

### 📖 Introduction

This lab teaches you how to construct a Knowledge Graph from structured data and implement Retrieval-Augmented Generation (RAG) for intelligent query answering.

### 🎯 Learning Objectives

By completing this lab, you will:
1. Load and process triple-based knowledge data
2. Construct directed graphs using NetworkX
3. Visualize knowledge graphs with Matplotlib and Plotly
4. Implement graph sampling and filtering techniques
5. Build a RAG system using knowledge graphs and GPT-2
6. Query knowledge graphs for information retrieval
7. Generate contextual insights using language models

---

### 📂 Dataset Description

**Source**: DBpedia RDF triples

**Format**: CSV with three columns:
- **subject**: Entity (e.g., "Liu Chao-shiuan")
- **predicate**: Entity type/category (e.g., "Politician")
- **object**: Specific classification (e.g., "PrimeMinister")

**Structure**: Subject-Predicate-Object triples representing facts

**Expected Input**: `dbpedia.csv` file in the working directory

---

### 📊 Task 1: Load Data and Create Knowledge Graph

#### Objective
Transform triple data into a graph structure where entities are nodes and relationships are edges.

#### Instructions

```python
import pandas as pd
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

# Load the filtered dataset
dbpedia_path = "dbpedia.csv"
df = pd.read_csv(dbpedia_path, names=['subject', 'predicate', 'object'])

# Display first few rows
print(df.head())
print(f"Dataset shape: {df.shape}")

# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
for _, row in df.iterrows():
    subject = row['subject']
    predicate = row['predicate']
    obj = row['object']
    
    # Add entities (nodes) and relationships (edges)
    G.add_node(subject)
    G.add_node(obj)
    G.add_edge(subject, obj, relationship=predicate)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
```

#### 💡 Key Concepts
- **Directed Graph**: Edges have direction (A → B ≠ B → A)
- **Nodes**: Entities (subjects and objects)
- **Edges**: Relationships with attributes (predicates)
- **Graph Properties**: Number of nodes, edges, connectivity

---

### 🎨 Task 2: Visualize the Knowledge Graph (Static)

#### Objective
Create a static visualization of the entire knowledge graph using Matplotlib.

#### Instructions

```python
import matplotlib.pyplot as plt

# Define layout for nodes (spring layout for force-directed positioning)
pos = nx.spring_layout(G, k=0.15, iterations=20)

# Create figure
fig1 = plt.figure(figsize=(12, 12))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)

# Draw edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

plt.title("Knowledge Graph Visualization", fontsize=16)
plt.axis('off')  # Turn off axes
plt.tight_layout()
plt.show()
```

#### 📊 Visualization Components
- **Blue Nodes**: Entities from the knowledge graph
- **Gray Edges**: Relationships between entities
- **Spring Layout**: Nodes repel each other, edges act as springs
- **Label Size**: Adjusted for readability

#### 💡 Layout Algorithms
- **spring_layout**: Force-directed (good for general graphs)
- **circular_layout**: Nodes in a circle
- **kamada_kawai_layout**: Another force-directed option
- **spectral_layout**: Based on graph spectrum

---

### 🔍 Task 3: Interactive Visualization with Plotly

#### Objective
Create an interactive, filterable visualization of a knowledge graph subgraph.

#### Why Sample the Graph?
- Large graphs are cluttered and slow to render
- Focus on interesting substructures
- Interactive exploration is more effective on smaller subgraphs

#### Instructions

```python
import plotly.graph_objects as go
import random

# Sample 150 random nodes
sampled_nodes = random.sample(list(G.nodes()), 150)
subgraph = G.subgraph(sampled_nodes)

# Calculate positions
pos = nx.spring_layout(subgraph)

# Extract edge coordinates
edge_x = []
edge_y = []
for edge in subgraph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])  # None creates a break
    edge_y.extend([y0, y1, None])

# Create edge trace
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Extract node coordinates and properties
node_x = [pos[node][0] for node in subgraph.nodes()]
node_y = [pos[node][1] for node in subgraph.nodes()]
node_text = list(subgraph.nodes())
node_degree = [subgraph.degree(n) for n in subgraph.nodes()]

# Create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        color=node_degree,  # Color by degree
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2
    )
)

# Create figure
fig2 = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title='Interactive Knowledge Graph Subgraph',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)

# Show interactive plot
fig2.show()
```

#### 📊 Interactive Features
- **Hover**: See entity names
- **Color Coding**: Node degree (number of connections)
- **Zoom/Pan**: Explore different areas
- **Click & Drag**: Rearrange nodes

#### 💡 Key Concepts
- **Node Degree**: Number of connections (high degree = hub)
- **Subgraph**: A subset of the original graph
- **Color Scale**: Visual encoding of quantitative data
- **Interactive Visualization**: Better for exploration than static images

---

### 🤖 Task 4: Implement RAG with Knowledge Graph

#### Objective
Build a Retrieval-Augmented Generation system that retrieves information from the knowledge graph and generates insights using GPT-2.

#### What is RAG?
**Retrieval-Augmented Generation** combines:
1. **Retrieval**: Finding relevant information from a knowledge source
2. **Generation**: Using an LLM to synthesize information into answers

#### Architecture
```
User Query → Retrieve from KG → Add to Prompt → GPT-2 → Generated Answer
```

#### Instructions

```python
import pandas as pd
from transformers import pipeline

# Convert first 50 records to knowledge graph format
first_50_records = df.head(50)
knowledge_graph = first_50_records[['subject', 'predicate', 'object']].to_dict(orient='records')

# Retrieval function
def retrieve(query, knowledge_graph):
    """
    Search knowledge graph for entities matching the query
    """
    for record in knowledge_graph:
        if query.lower() in record["subject"].lower():
            return (f"Subject: {record['subject']}, "
                   f"Predicate: {record['predicate']}, "
                   f"Object: {record['object']}")
    return "Entity not found in the Knowledge Graph."

# Initialize GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# Generation function
def generate_insight(retrieved_data):
    """
    Generate insights using GPT-2 based on retrieved data
    """
    prompt = (f"The following information was retrieved from the knowledge graph: "
             f"{retrieved_data}. Based on this information, generate an insightful "
             f"response.\n")
    
    return generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']

# Full RAG pipeline
def process_query(query, knowledge_graph):
    """
    Complete RAG process: retrieve then generate
    """
    retrieved_data = retrieve(query, knowledge_graph)
    
    if retrieved_data != "Entity not found in the Knowledge Graph":
        insight = generate_insight(retrieved_data)
        return f"Answer to question '{query}': {insight}"
    else:
        return f"Answer to question '{query}': {retrieved_data}"

# Example usage
if __name__ == "__main__":
    query = "Hohnstein Castle"
    answer = process_query(query, knowledge_graph)
    print(answer)
```

#### 📊 Example Output

**Query**: "Hohnstein Castle"

**Retrieved**:
```
Subject: Hohnstein Castle, Predicate: Building, Object: Castle
```

**Generated**:
```
Answer to question 'Hohnstein Castle': The following information was retrieved 
from the knowledge graph: Subject: Hohnstein Castle, Predicate: Building, 
Object: Castle. Based on this information, generate an insightful response.

Hohnstein Castle is a historic building classified as a castle. This medieval 
structure likely served defensive and residential purposes, and represents 
significant architectural heritage...
```

#### 💡 Key Concepts

##### Retrieval Phase
- **Exact Match**: Case-insensitive substring matching
- **Structured Data**: Returns subject-predicate-object triple
- **Fallback**: Returns "not found" message if no match

##### Generation Phase
- **Prompt Engineering**: Template that provides context
- **Max Length**: Controls output length (300 tokens)
- **Temperature**: Controls randomness (default in pipeline)

##### RAG Benefits
- **Factual Grounding**: Answers based on real data
- **Reduced Hallucination**: LLM works with provided facts
- **Explainable**: Can trace answer back to knowledge graph
- **Up-to-date**: Update KG without retraining LLM

---

### 🔧 Advanced RAG Techniques

#### Improvement Strategies

**1. Better Retrieval**
```python
def fuzzy_retrieve(query, knowledge_graph, threshold=80):
    from fuzzywuzzy import fuzz
    
    best_match = None
    best_score = 0
    
    for record in knowledge_graph:
        score = fuzz.partial_ratio(query.lower(), record["subject"].lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = record
    
    if best_match:
        return format_triple(best_match)
    return "Entity not found in the Knowledge Graph."
```

**2. Multi-hop Retrieval**
```python
def multi_hop_retrieve(query, graph, max_hops=2):
    """Retrieve information from neighboring nodes"""
    # Find starting node
    start_node = find_node(query, graph)
    
    # Get neighbors within max_hops
    neighbors = nx.single_source_shortest_path_length(
        graph, start_node, cutoff=max_hops
    )
    
    # Collect information from neighborhood
    context = []
    for node in neighbors:
        context.extend(graph.edges(node, data=True))
    
    return context
```

**3. Better Prompts**
```python
def generate_with_context(retrieved_data, query):
    prompt = f"""You are a helpful assistant with access to a knowledge graph.

User Question: {query}

Relevant Information from Knowledge Graph:
{retrieved_data}

Please provide a comprehensive answer based on the above information. 
If the information is insufficient, acknowledge what is known and what is unknown.

Answer:"""
    
    return generator(prompt, max_length=500, temperature=0.7)
```

---

## 💻 Installation Guide

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Gen AI E2 Hands on Lab"
```

### Step 2: Set Up Python Environment

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n genai-lab python=3.8

# Activate environment
conda activate genai-lab
```

### Step 3: Install Dependencies

```bash
# Install core libraries
pip install pandas numpy scikit-learn nltk networkx matplotlib seaborn plotly transformers

# Or use requirements file (if provided)
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 5: Prepare Data Files

**For Lab 1**: Place `train.csv` in the project directory

**For Lab 2**: Place `dbpedia.csv` in the project directory

### Step 6: Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: NLTK Data Not Found
```
Error: Resource stopwords not found
```

**Solution**:
```python
import nltk
nltk.download('all')  # Downloads all NLTK data
```

#### Issue 2: Memory Error with Large Knowledge Graph
```
MemoryError: Unable to allocate array
```

**Solution**:
```python
# Sample the data before creating graph
df_sample = df.sample(n=10000)  # Use 10,000 records instead of all
```

#### Issue 3: Transformer Model Too Large
```
Error: Model size exceeds available memory
```

**Solution**:
```python
# Use DistilGPT2 (smaller, faster)
generator = pipeline('text-generation', model='distilgpt2')
```

#### Issue 4: Plotly Not Displaying in Jupyter
```python
# Install Jupyter extensions
pip install "notebook>=5.3" "ipywidgets>=7.2"
jupyter nbextension enable --py widgetsnbextension
```

#### Issue 5: Slow Training Time
**Solution**:
- Reduce max_features in TfidfVectorizer: `TfidfVectorizer(max_features=5000)`
- Reduce n_estimators in Random Forest: `RandomForestClassifier(n_estimators=50)`
- Use a smaller subset of data for initial experiments

---

## 📈 Learning Outcomes

### After Completing Lab 1, You Will Be Able To:
✅ Load and explore real-world textual datasets  
✅ Handle missing data appropriately  
✅ Perform comprehensive text preprocessing  
✅ Apply lemmatization and stopword removal  
✅ Create document-term matrices  
✅ Implement TF-IDF vectorization  
✅ Split data into training and test sets  
✅ Train Random Forest classifiers  
✅ Interpret classification metrics  
✅ Visualize model performance with confusion matrices  

### After Completing Lab 2, You Will Be Able To:
✅ Load and process RDF triple data  
✅ Construct directed graphs from structured data  
✅ Visualize graphs using Matplotlib and Plotly  
✅ Sample and filter large graphs  
✅ Implement knowledge graph querying  
✅ Build RAG systems combining retrieval and generation  
✅ Use transformer models for text generation  
✅ Design prompts for language models  
✅ Integrate external knowledge with LLMs  

---

## 📚 Additional Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [NLTK Book](https://www.nltk.org/book/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Plotly Python](https://plotly.com/python/)

### Tutorials
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Random Forests - StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Knowledge Graphs Introduction](https://www.ontotext.com/knowledgehub/fundamentals/what-is-a-knowledge-graph/)
- [RAG Explained](https://arxiv.org/abs/2005.11401)

### Datasets
- [Kaggle NLP Datasets](https://www.kaggle.com/datasets?tags=13204-NLP)
- [DBpedia](https://www.dbpedia.org/)
- [Wikidata](https://www.wikidata.org/)

### Advanced Topics
- **Active Learning**: Improve models with selective labeling
- **BERT for Classification**: Use transformer-based models
- **Graph Neural Networks**: Deep learning on graphs
- **Vector Databases**: Scale RAG systems (Pinecone, Weaviate)
- **LangChain**: Framework for building RAG applications

---

## 🎓 Assessment Criteria

### Lab 1 Evaluation
| Task | Points | Criteria |
|------|--------|----------|
| Data Loading | 10 | Correct loading, missing value handling |
| Text Cleaning | 25 | Complete preprocessing pipeline |
| Count Vectorizer | 15 | Correct document-term matrix |
| TF-IDF | 20 | Proper vectorization, train-test split |
| Classification | 30 | Model training, evaluation, interpretation |

### Lab 2 Evaluation
| Task | Points | Criteria |
|------|--------|----------|
| Graph Creation | 20 | Correct graph structure |
| Static Visualization | 15 | Clear, labeled visualization |
| Interactive Plot | 20 | Proper Plotly implementation |
| RAG System | 45 | Retrieval, generation, integration |

---

## 🤝 Contributing

If you're an instructor or trainer using this material:

1. **Fork** this repository
2. **Add** your own examples or variations
3. **Share** improvements back to the community
4. **Customize** for your specific training context

---

## 📝 License

This training material is provided for educational purposes. Please cite appropriately if used in academic or professional settings.

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Additional Resources](#additional-resources)
3. Open an issue in the repository
4. Contact your instructor or training coordinator

---

## 🎯 Next Steps

After completing these labs, consider:

1. **Advanced NLP**: Explore BERT, GPT, and transformer architectures
2. **Production Systems**: Learn about MLOps and model deployment
3. **Specialized Domains**: Apply techniques to domain-specific problems
4. **Research**: Read recent papers on RAG, Knowledge Graphs, and NLP
5. **Projects**: Build end-to-end applications using these techniques

---

**Happy Learning! 🚀**

*Last Updated: December 2025*
