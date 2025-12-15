# Quick Reference Guide

## 🚀 Getting Started in 5 Minutes

### 1. Setup Environment
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install packages
pip install pandas numpy scikit-learn nltk networkx matplotlib seaborn plotly transformers
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

---

## 📊 Lab 1 Cheat Sheet: Tweet Classification

### Essential Imports
```python
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### Task Checklist
- [ ] **Task 1**: Load `train.csv`, drop `keyword` and `location` columns
- [ ] **Task 2**: Clean text (URLs, HTML, stopwords, lemmatization, emojis)
- [ ] **Task 3**: Create document-term matrix with `CountVectorizer`
- [ ] **Task 4**: Create TF-IDF vectors with 80/20 train-test split
- [ ] **Task 5**: Train Random Forest, evaluate with F1-score and confusion matrix

### Key Variables to Remember
```python
data_cleaned      # Task 1: DataFrame after dropping columns
clean_df          # Task 2: First 5 rows after cleaning
sample_df         # Task 3: Document-term matrix
tfidf_vectorizer  # Task 4: TF-IDF vectorizer object
tfidf_train_vectors, tfidf_test_vectors  # Task 4: Transformed vectors
classifier        # Task 5: Trained Random Forest model
f1score           # Task 5: F1-score value
```

### Common Text Cleaning Pattern
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)     # Remove HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Keep only letters
    words = [w for w in text.split() if w not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)
```

### Train-Test Split Template
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values,
    df['target'].values,
    test_size=0.2,
    random_state=123,
    stratify=df['target'].values
)
```

---

## 🕸️ Lab 2 Cheat Sheet: Knowledge Graph

### Essential Imports
```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import pipeline
import random
```

### Task Checklist
- [ ] **Task 1**: Load `dbpedia.csv`, create NetworkX DiGraph
- [ ] **Task 2**: Visualize full graph with Matplotlib
- [ ] **Task 3**: Create interactive subgraph with Plotly
- [ ] **Task 4**: Implement RAG system (retrieve + generate)

### Key Variables to Remember
```python
df                # DataFrame with subject, predicate, object
G                 # NetworkX DiGraph
fig1              # Matplotlib figure (static visualization)
edge_trace        # Plotly edge trace for interactive plot
node_trace        # Plotly node trace for interactive plot
fig2              # Plotly figure (interactive visualization)
generator         # GPT-2 pipeline for text generation
answer            # Generated answer from RAG system
knowledge_graph   # List of dictionaries (first 50 records)
```

### Creating a Graph
```python
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_node(row['subject'])
    G.add_node(row['object'])
    G.add_edge(row['subject'], row['object'], relationship=row['predicate'])
```

### RAG Pattern
```python
# 1. Retrieve
def retrieve(query, knowledge_graph):
    for record in knowledge_graph:
        if query.lower() in record["subject"].lower():
            return f"Subject: {record['subject']}, Predicate: {record['predicate']}, Object: {record['object']}"
    return "Entity not found in the Knowledge Graph."

# 2. Generate
generator = pipeline('text-generation', model='gpt2')

def generate_insight(retrieved_data):
    prompt = f"The following information was retrieved from the knowledge graph: {retrieved_data}. Based on this information, generate an insightful response.\n"
    return generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']

# 3. Combine
def process_query(query, knowledge_graph):
    retrieved_data = retrieve(query, knowledge_graph)
    if retrieved_data != "Entity not found in the Knowledge Graph":
        return generate_insight(retrieved_data)
    return retrieved_data
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install <package-name>` |
| NLTK stopwords error | Run `nltk.download('stopwords')` |
| Memory error | Reduce dataset size: `df = df.sample(n=10000)` |
| Plotly not showing | Install extensions: `pip install "notebook>=5.3" "ipywidgets>=7.2"` |
| Training too slow | Reduce features: `TfidfVectorizer(max_features=5000)` |
| GPT-2 too large | Use smaller model: `pipeline('text-generation', model='distilgpt2')` |

---

## 📈 Expected Outputs

### Lab 1
- **DataFrame shape**: ~7600 rows × 3 columns (after dropping keyword & location)
- **TF-IDF features**: ~11,000-14,000 features
- **F1-Score**: ~0.70-0.78
- **Accuracy**: ~0.75-0.82

### Lab 2
- **Graph nodes**: Depends on dataset size
- **Graph edges**: Equal to number of triples
- **Subgraph**: 150 nodes (as specified in sampling)
- **RAG output**: ~200-300 words

---

## 💡 Pro Tips

### Lab 1 Tips
1. **Always lowercase** text before processing
2. **Fit on train, transform on test** - never fit on test data
3. **Stratify** your split to maintain class balance
4. **Check vocabulary size** - should be ~11,000-14,000 features
5. **F1-score** is better than accuracy for imbalanced datasets

### Lab 2 Tips
1. **Sample large graphs** - full visualization can be overwhelming
2. **Use spring_layout** for general graphs (force-directed)
3. **Color by degree** - helps identify hub nodes
4. **Limit GPT-2 output** - use `max_length` to control response size
5. **Prompt engineering matters** - clear prompts = better responses

---

## 🎯 Key Formulas

### TF-IDF
```
TF-IDF(word, doc) = TF(word, doc) × log(N / df(word))

where:
- TF(word, doc) = Count of word in document
- N = Total number of documents
- df(word) = Number of documents containing word
```

### Classification Metrics
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Node Degree (Knowledge Graph)
```
Degree(node) = Number of edges connected to node

In directed graphs:
- In-degree = Incoming edges
- Out-degree = Outgoing edges
```

---

## 📝 Testing Your Variables

### Lab 1 Test Commands
```python
# Task 1
assert 'keyword' not in data_cleaned.columns
assert 'location' not in data_cleaned.columns

# Task 2
assert clean_df.shape[0] == 5  # First 5 rows

# Task 3
assert sample_df.shape[0] == 2  # 2 documents

# Task 4
assert tfidf_train_vectors.shape[0] == len(X_train)
assert tfidf_test_vectors.shape[0] == len(X_test)
assert 11000 <= tfidf_train_vectors.shape[1] <= 14000

# Task 5
assert 0 <= f1score <= 1
assert classifier is not None
```

### Lab 2 Test Commands
```python
# Task 1
assert G.number_of_nodes() > 0
assert G.number_of_edges() > 0
assert nx.is_directed(G)

# Task 3
assert len(subgraph.nodes()) == 150

# Task 4
assert generator is not None
assert isinstance(answer, str)
assert len(knowledge_graph) == 50
```

---

## 🔗 Quick Links

- **Main Documentation**: [README.md](README.md)
- **Scikit-learn**: https://scikit-learn.org/
- **NLTK**: https://www.nltk.org/
- **NetworkX**: https://networkx.org/
- **Hugging Face**: https://huggingface.co/
- **Plotly**: https://plotly.com/python/

---

## ⏱️ Time Estimates

| Lab | Task | Estimated Time |
|-----|------|----------------|
| Lab 1 | Task 1: Data Loading | 5-10 min |
| Lab 1 | Task 2: Text Cleaning | 15-20 min |
| Lab 1 | Task 3: Count Vectorizer | 10-15 min |
| Lab 1 | Task 4: TF-IDF | 10-15 min |
| Lab 1 | Task 5: Classification | 15-20 min |
| **Lab 1 Total** | | **~60-90 min** |
| Lab 2 | Task 1: Graph Creation | 10-15 min |
| Lab 2 | Task 2: Static Viz | 5-10 min |
| Lab 2 | Task 3: Interactive Viz | 15-20 min |
| Lab 2 | Task 4: RAG System | 20-30 min |
| **Lab 2 Total** | | **~50-75 min** |

---

**Print this for easy reference during labs! 📄**
