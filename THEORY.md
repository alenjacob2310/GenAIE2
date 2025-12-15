# Theoretical Background & Concepts

## 📚 Table of Contents
1. [Natural Language Processing Fundamentals](#nlp-fundamentals)
2. [Text Vectorization Methods](#text-vectorization)
3. [Machine Learning for Text Classification](#ml-classification)
4. [Knowledge Graphs](#knowledge-graphs)
5. [Retrieval-Augmented Generation](#rag)
6. [Evaluation Metrics](#evaluation-metrics)

---

## 🔤 NLP Fundamentals

### What is Natural Language Processing?

**Natural Language Processing (NLP)** is a field of AI focused on enabling computers to understand, interpret, and generate human language.

#### Key Challenges in NLP:
1. **Ambiguity**: Words have multiple meanings (e.g., "bank" = financial institution or river bank)
2. **Context Dependency**: Meaning changes with context
3. **Informal Language**: Slang, typos, abbreviations
4. **Cultural Nuances**: Idioms, sarcasm, cultural references

### Text Preprocessing Pipeline

```
Raw Text
    ↓
Lowercasing ("The Cat" → "the cat")
    ↓
URL/HTML Removal
    ↓
Punctuation Removal ("Hello!" → "Hello")
    ↓
Tokenization ("Hello world" → ["Hello", "world"])
    ↓
Stopword Removal (["the", "cat", "is"] → ["cat"])
    ↓
Lemmatization (["running", "ran"] → ["run", "run"])
    ↓
Clean Text
```

### Stopwords

**Definition**: Common words that carry little semantic meaning.

**Examples**: the, is, at, which, on, a, an

**Why Remove?**
- Reduce noise in the data
- Decrease feature space dimensionality
- Focus on meaningful content words

**When NOT to Remove?**
- Sentiment analysis ("not good" vs "good")
- Named entity recognition
- Question answering systems

### Stemming vs. Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Method** | Rule-based truncation | Dictionary-based |
| **Output** | May not be real word | Always valid word |
| **Speed** | Faster | Slower |
| **Accuracy** | Lower | Higher |
| **Example** | "studies" → "studi" | "studies" → "study" |

**Example Comparison**:
```
Word: "better"
Stemming: "better" → "bet"  (incorrect)
Lemmatization: "better" → "good"  (correct)

Word: "running"
Stemming: "running" → "run"
Lemmatization: "running" → "run"  (both correct here)
```

---

## 📊 Text Vectorization

### The Fundamental Problem

**Computers work with numbers, but language is symbolic.**

We need to convert text → numerical vectors for machine learning.

### Method 1: Bag of Words (BoW)

**Concept**: Represent text as a collection of words, ignoring grammar and order.

**Process**:
1. Create vocabulary of all unique words
2. Count occurrences of each word in each document
3. Create a matrix: rows = documents, columns = words

**Example**:
```
Documents:
Doc1: "cat sat on mat"
Doc2: "dog sat on floor"

Vocabulary: [cat, dog, floor, mat, on, sat]

BoW Matrix:
        cat  dog  floor  mat  on  sat
Doc1     1    0     0     1   1    1
Doc2     0    1     1     0   1    1
```

**Advantages**:
✅ Simple to understand and implement
✅ Works well for many applications
✅ Fast to compute

**Disadvantages**:
❌ Loses word order ("dog bites man" = "man bites dog")
❌ Doesn't capture semantics
❌ Large vocabulary = high dimensionality
❌ Treats all words equally (common words dominate)

### Method 2: TF-IDF

**TF-IDF** = Term Frequency × Inverse Document Frequency

**Purpose**: Weight words by importance, not just frequency.

#### Term Frequency (TF)
```
TF(word, document) = (# times word appears in document) / (total words in document)
```

**Intuition**: How important is this word in THIS document?

#### Inverse Document Frequency (IDF)
```
IDF(word) = log(Total # of documents / # documents containing word)
```

**Intuition**: How rare is this word across ALL documents?

#### Combined TF-IDF
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)
```

**Example**:

Corpus:
```
Doc1: "cat cat dog"
Doc2: "dog dog dog"
Doc3: "cat fish"
```

For word "cat" in Doc1:
```
TF(cat, Doc1) = 2/3 = 0.667
IDF(cat) = log(3/2) = 0.176  (appears in 2 out of 3 docs)
TF-IDF(cat, Doc1) = 0.667 × 0.176 = 0.117
```

For word "dog" in Doc1:
```
TF(dog, Doc1) = 1/3 = 0.333
IDF(dog) = log(3/2) = 0.176  (appears in 2 out of 3 docs)
TF-IDF(dog, Doc1) = 0.333 × 0.176 = 0.059
```

**Why TF-IDF is Better than BoW**:
- Downweights common words (appearing in many documents)
- Upweights rare but meaningful words
- Better captures document uniqueness

### Method 3: Word Embeddings (Advanced)

**Examples**: Word2Vec, GloVe, FastText

**Concept**: Map words to dense vectors where semantically similar words are close.

**Example**:
```
Vector("king") - Vector("man") + Vector("woman") ≈ Vector("queen")
```

**Not covered in these labs but important to know for future learning.**

---

## 🤖 Machine Learning for Text Classification

### The Classification Task

**Goal**: Given text, predict a category (e.g., spam/not spam, positive/negative, disaster/not disaster)

### Training Process

```
Training Data (labeled examples)
    ↓
Feature Extraction (TF-IDF)
    ↓
Model Training (Random Forest)
    ↓
Trained Model
    ↓
Prediction on New Data
```

### Why Random Forest?

**Random Forest** = Ensemble of multiple decision trees

#### How it Works:
1. **Bootstrap Sampling**: Create N random subsets of training data
2. **Build Trees**: Train a decision tree on each subset
3. **Random Features**: Each tree uses random subset of features
4. **Voting**: Final prediction = majority vote from all trees

#### Advantages for Text Classification:
✅ Handles high-dimensional data (many features)
✅ Resistant to overfitting
✅ Provides feature importance
✅ Works well with sparse data (TF-IDF matrices)
✅ No need for feature scaling
✅ Can handle non-linear relationships

#### Decision Tree Example:

```
                [Is "fire" present?]
                /                \
              Yes                 No
              /                     \
      [Is "emergency"          [Is "weather"
       present?]                present?]
        /      \                 /        \
      Yes      No              Yes        No
      /         \              /           \
  DISASTER   NOT DISASTER   DISASTER   NOT DISASTER
```

**Random Forest** = Combine 100+ such trees with different features and data.

### Training vs. Testing

#### Why Split Data?

**Problem**: A model that memorizes training data won't generalize to new data.

**Solution**: Hold out test data to evaluate generalization.

**Common Split**: 80% training, 20% testing

#### Stratification

**Definition**: Ensure train and test sets have same class distribution.

**Example**:
```
Original Data: 70% Not Disaster, 30% Disaster

Without Stratification (random):
Train: 65% Not Disaster, 35% Disaster  ⚠️
Test: 80% Not Disaster, 20% Disaster   ⚠️

With Stratification:
Train: 70% Not Disaster, 30% Disaster  ✅
Test: 70% Not Disaster, 30% Disaster   ✅
```

---

## 🕸️ Knowledge Graphs

### What is a Knowledge Graph?

**Definition**: A structured representation of knowledge as entities (nodes) and relationships (edges).

### Structure: RDF Triples

**Format**: Subject - Predicate - Object

**Examples**:
```
(Albert Einstein, is-a, Physicist)
(Albert Einstein, born-in, Ulm)
(Ulm, located-in, Germany)
(Germany, is-a, Country)
```

### Graph Representation

```
    [Albert Einstein] --is-a--> [Physicist]
            |
         born-in
            |
            ↓
         [Ulm] --located-in--> [Germany] --is-a--> [Country]
```

### Types of Graphs

#### Undirected Graph
```
A --- B
|     |
C --- D
```
Edges have no direction. Relationship is symmetric.

#### Directed Graph (Used in Our Lab)
```
A --> B
|     ↑
↓     |
C --> D
```
Edges have direction. Relationship is asymmetric.

### Graph Properties

#### Node Degree
**Definition**: Number of edges connected to a node.

```
    A
   / \
  B   C
   \ /
    D
    
Degree(A) = 2
Degree(B) = 2
Degree(C) = 2
Degree(D) = 2
```

In **directed graphs**:
- **In-degree**: # of incoming edges
- **Out-degree**: # of outgoing edges

#### Hub Nodes
**Definition**: Nodes with high degree (many connections)

**Example**: In a social network, celebrities are hubs.

### Graph Layouts

**Spring Layout** (Force-Directed):
- Nodes repel each other
- Edges act as springs pulling connected nodes together
- Iteratively finds equilibrium
- Good for general graphs

**Circular Layout**:
- Nodes arranged in a circle
- Good for showing overall connectivity

**Hierarchical Layout**:
- Nodes arranged in levels
- Good for trees and DAGs

### Knowledge Graph Use Cases

1. **Search Engines**: Google Knowledge Graph
2. **Question Answering**: Answering "Who is X?" queries
3. **Recommendation Systems**: Finding related items
4. **Drug Discovery**: Linking diseases, drugs, genes
5. **Fraud Detection**: Finding suspicious relationship patterns

---

## 🧠 Retrieval-Augmented Generation (RAG)

### The Problem with Pure LLMs

**Large Language Models** (GPT, BERT, etc.) have limitations:
1. **Static Knowledge**: Trained on past data, not up-to-date
2. **Hallucination**: Generate plausible-sounding but false information
3. **No Sources**: Can't cite where information came from
4. **Domain Specificity**: May lack specialized knowledge

### The RAG Solution

**RAG** = Retrieval-Augmented Generation

**Concept**: Combine retrieval from knowledge source + generation from LLM

### RAG Architecture

```
                    User Query
                        |
                        ↓
                  [Retrieval System]
                        |
                        ↓
                [Knowledge Source]
                (Knowledge Graph, Database, Documents)
                        |
                        ↓
                 [Retrieved Context]
                        |
                        ↓
            Query + Context → [LLM]
                        |
                        ↓
                  Generated Answer
```

### Example Flow

**Query**: "What is Hohnstein Castle?"

**Step 1 - Retrieve**:
```sql
Search Knowledge Graph for "Hohnstein Castle"
→ Result: (Hohnstein Castle, Building, Castle)
```

**Step 2 - Create Prompt**:
```
Context: Hohnstein Castle is classified as a Building of type Castle.
Question: What is Hohnstein Castle?
Answer:
```

**Step 3 - Generate**:
```
Hohnstein Castle is a historic castle, classified as a building 
structure. Castles of this type typically served as fortified 
residences during medieval times...
```

### Benefits of RAG

✅ **Factual Grounding**: Answers based on real data
✅ **Up-to-date**: Update knowledge source without retraining LLM
✅ **Source Attribution**: Can cite where information came from
✅ **Domain Expertise**: Add specialized knowledge easily
✅ **Reduced Hallucination**: LLM works with provided facts
✅ **Cost-Effective**: Don't need to retrain large models

### RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Update Knowledge** | Update knowledge source | Retrain model |
| **Cost** | Low | High (GPU, time) |
| **Explainability** | Can trace to source | Black box |
| **Specialization** | Easy to add domains | Requires training data |
| **Response Time** | Slightly slower | Fast |

### Advanced RAG Techniques

#### 1. Dense Retrieval
Use embeddings instead of keyword matching:
```python
# Convert query to embedding
query_embedding = embed(query)

# Find similar documents by embedding similarity
similar_docs = find_nearest(query_embedding, doc_embeddings)
```

#### 2. Multi-Hop Reasoning
Retrieve multiple related facts:
```
Query: "Who invented the telephone?"
Hop 1: Retrieve → Alexander Graham Bell invented telephone
Hop 2: Retrieve → Bell was born in Scotland
Hop 3: Retrieve → Bell moved to America
Combined Answer: Alexander Graham Bell, a Scottish inventor who 
                 moved to America, invented the telephone.
```

#### 3. Re-Ranking
Retrieve many candidates, then re-rank by relevance:
```python
candidates = retrieve_top_100(query)
ranked = rerank_by_relevance(query, candidates)
top_k = ranked[:5]
```

---

## 📈 Evaluation Metrics

### Confusion Matrix

**2×2 Matrix for Binary Classification**:

|                    | Predicted Positive | Predicted Negative |
|--------------------|--------------------|--------------------|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

**Example - Disaster Tweet Classification**:

|                    | Predicted Disaster | Predicted Not Disaster |
|--------------------|-------------------|------------------------|
| **Actually Disaster** | 150 (TP) | 50 (FN) |
| **Actually Not Disaster** | 30 (FP) | 270 (TN) |

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (150 + 270) / 500
         = 0.84 = 84%
```

**Interpretation**: Correct predictions out of all predictions.

**Problem with Accuracy**:
If 95% of tweets are not disasters:
- Predict "Not Disaster" for everything → 95% accuracy!
- But model is useless (never detects disasters)

### Precision

```
Precision = TP / (TP + FP)
          = 150 / (150 + 30)
          = 0.833 = 83.3%
```

**Question Answered**: "Of all tweets we predicted as disasters, how many were actually disasters?"

**High Precision**: Few false alarms

**Use Case**: Medical diagnosis (avoid false positives)

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
       = 150 / (150 + 50)
       = 0.75 = 75%
```

**Question Answered**: "Of all actual disasters, how many did we catch?"

**High Recall**: Don't miss actual disasters

**Use Case**: Security screening (don't miss threats)

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.833 × 0.75) / (0.833 + 0.75)
   = 0.789 = 78.9%
```

**Interpretation**: Harmonic mean of Precision and Recall.

**When to Use**: When you want balance between Precision and Recall.

### Precision-Recall Tradeoff

**Threshold Adjustment**:

```
Probability → Prediction
0.9 → Disaster     High Precision, Low Recall
0.7 → Disaster     Balanced
0.3 → Disaster     Low Precision, High Recall
```

**Visual Example**:

```
Conservative Model (High Precision):
Only predict disaster when very confident
→ Few false alarms, but miss some disasters

Aggressive Model (High Recall):
Predict disaster with low threshold
→ Catch most disasters, but many false alarms
```

### Which Metric to Use?

| Scenario | Prioritize | Reason |
|----------|-----------|--------|
| Spam Detection | Precision | Don't want important emails in spam |
| Cancer Screening | Recall | Don't miss actual cancer cases |
| Fraud Detection | Recall | Catch all fraud, false alarms acceptable |
| News Classification | F1-Score | Balance both aspects |

---

## 🎓 Summary of Key Concepts

### NLP Pipeline
1. **Clean** → Remove noise (URLs, HTML, punctuation)
2. **Tokenize** → Split into words
3. **Normalize** → Lowercase, lemmatize
4. **Vectorize** → Convert to numbers (TF-IDF)
5. **Model** → Train classifier
6. **Evaluate** → Test on unseen data

### Knowledge Representation
1. **Entities** → Nodes in graph
2. **Relationships** → Edges in graph
3. **Triples** → (Subject, Predicate, Object)
4. **Queries** → Retrieve relevant information
5. **Reasoning** → Infer new knowledge

### RAG Process
1. **Query** → User asks a question
2. **Retrieve** → Find relevant facts
3. **Augment** → Add facts to prompt
4. **Generate** → LLM creates answer
5. **Return** → Present answer to user

### Model Evaluation
1. **Split Data** → Train/Test (80/20)
2. **Train** → Learn from training data
3. **Predict** → Apply to test data
4. **Metrics** → Calculate Precision, Recall, F1
5. **Interpret** → Understand strengths/weaknesses

---

## 📚 Further Reading

### Books
- **"Speech and Language Processing"** by Jurafsky & Martin
- **"Natural Language Processing with Python"** by Bird, Klein & Loper
- **"Hands-On Machine Learning"** by Aurélien Géron

### Research Papers
- **TF-IDF**: Salton & McGill (1983) "Introduction to Modern Information Retrieval"
- **Random Forests**: Breiman (2001) "Random Forests"
- **RAG**: Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

### Online Courses
- Stanford CS224N: Natural Language Processing with Deep Learning
- Fast.ai: Practical Deep Learning for Coders
- Coursera: NLP Specialization by deeplearning.ai

---

**Use this document as your theoretical foundation for the hands-on labs!**

*Last Updated: December 2025*
