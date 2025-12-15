# Instructor Guide

## 📋 Overview

This guide provides instructors with teaching strategies, common student challenges, assessment rubrics, and extension activities for the Gen AI E2 Hands-on Labs.

---

## 🎯 Learning Objectives

### Lab 1: NLP Disaster Tweets Prediction
**Knowledge Level (Remember/Understand)**:
- Define TF-IDF and explain its advantages over simple word counts
- Describe the purpose of text preprocessing steps
- Explain the difference between training and testing data

**Application Level (Apply/Analyze)**:
- Apply text cleaning techniques to real-world social media data
- Construct document-term matrices and TF-IDF vectors
- Implement train-test splitting with stratification

**Evaluation/Creation Level**:
- Evaluate model performance using multiple metrics
- Interpret confusion matrices and classification reports
- Compare different preprocessing approaches

### Lab 2: Knowledge Graph & RAG
**Knowledge Level**:
- Define knowledge graphs and RDF triples
- Explain Retrieval-Augmented Generation
- Understand graph properties (nodes, edges, degree)

**Application Level**:
- Construct directed graphs from structured data
- Implement retrieval functions for knowledge graphs
- Integrate LLMs with external knowledge sources

**Evaluation/Creation Level**:
- Design effective retrieval strategies
- Optimize prompts for better generation
- Evaluate RAG system quality

---

## ⏰ Suggested Schedule

### Option 1: Two 2-Hour Sessions

**Session 1: Lab 1 - Tweet Classification**
- 0:00-0:15: Introduction to NLP and project overview
- 0:15-0:30: Task 1 walkthrough (data loading)
- 0:30-1:00: Task 2 guided practice (text preprocessing)
- 1:00-1:15: Break
- 1:15-1:30: Tasks 3-4 demonstration (vectorization)
- 1:30-2:00: Task 5 and evaluation discussion

**Session 2: Lab 2 - Knowledge Graphs**
- 0:00-0:15: Introduction to knowledge graphs
- 0:15-0:40: Tasks 1-2 (graph creation and visualization)
- 0:40-1:00: Task 3 (interactive visualization)
- 1:00-1:15: Break
- 1:15-1:45: Task 4 (RAG implementation)
- 1:45-2:00: Discussion and Q&A

### Option 2: One 4-Hour Workshop
- 0:00-0:20: Introduction to both labs
- 0:20-1:30: Lab 1 (with short breaks)
- 1:30-2:00: Lunch break
- 2:00-3:30: Lab 2 (with short breaks)
- 3:30-4:00: Wrap-up and extension activities

### Option 3: Self-Paced Online
Provide 1-2 weeks for completion with:
- Weekly check-in sessions (30 min)
- Discussion forum for questions
- Submission deadline for final notebooks

---

## 🎓 Teaching Strategies

### Pre-Lab Preparation

**One Week Before**:
1. Send installation instructions
2. Share pre-reading materials (THEORY.md)
3. Verify dataset availability
4. Test all code in fresh environment

**Day Before**:
1. Send reminder with checklist
2. Provide troubleshooting guide
3. Share Zoom/meeting link if remote

### During the Lab

**Introduction Phase (15 min)**:
- Show real-world applications
- Demonstrate expected outcomes
- Preview the workflow
- Address any setup issues

**Guided Practice**:
- Live code first task together
- Use "I do, we do, you do" approach
- Encourage questions
- Walk around (or use breakout rooms online)

**Independent Work**:
- Set clear time boxes for each task
- Provide hints, not solutions initially
- Use "productive struggle" - let students debug
- Check progress regularly

**Wrap-Up**:
- Review key concepts
- Share best solutions
- Preview next steps
- Assign any homework

### Common Teaching Pitfalls to Avoid

❌ **Going too fast** - Check understanding frequently
❌ **Providing solutions immediately** - Guide students to find answers
❌ **Skipping theory** - Explain why, not just how
❌ **Ignoring errors** - Use errors as teaching moments
❌ **Not adapting** - Adjust pace based on student needs

---

## 🐛 Common Student Challenges

### Lab 1 Challenges

#### Challenge 1: NLTK Data Not Downloaded
**Symptom**: `LookupError: Resource stopwords not found`

**Solution to Teach**:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Teaching Point**: External data dependencies must be explicitly downloaded.

#### Challenge 2: Regular Expressions Confusion
**Symptom**: Students don't understand regex patterns

**Teaching Strategy**:
- Show regex101.com for interactive testing
- Explain common patterns:
  - `\S+` = one or more non-whitespace
  - `[^a-zA-Z]` = anything not a letter
  - `<.*?>` = anything between angle brackets
- Practice with simple examples first

#### Challenge 3: Fit vs Transform
**Symptom**: Students fit on test data or don't understand why

**Teaching Analogy**:
```
Think of it like learning a language:

Training Phase: Learn the vocabulary (fit)
Testing Phase: Use that same vocabulary (transform)

You can't learn new words during the exam!
```

**Code Example**:
```python
# CORRECT ✅
vectorizer.fit(X_train)           # Learn vocabulary
train_vectors = vectorizer.transform(X_train)
test_vectors = vectorizer.transform(X_test)   # Use same vocabulary

# WRONG ❌
vectorizer.fit(X_test)            # Don't learn from test data!
```

#### Challenge 4: Understanding Metrics
**Symptom**: Confusion about when to use precision vs recall

**Teaching Strategy**:
- Use real-world scenarios:
  - **Airport Security** (High Recall): Don't miss threats
  - **Email Spam Filter** (High Precision): Don't lose important emails
- Draw confusion matrix on board
- Calculate metrics manually for small example

#### Challenge 5: Slow Training
**Symptom**: Code takes too long to run

**Solutions**:
```python
# Reduce vocabulary size
vectorizer = TfidfVectorizer(max_features=5000)

# Reduce number of trees
classifier = RandomForestClassifier(n_estimators=50)

# Use smaller dataset
df_sample = df.sample(n=5000)
```

### Lab 2 Challenges

#### Challenge 1: Graph Too Large to Visualize
**Symptom**: Matplotlib crashes or becomes unresponsive

**Teaching Point**: Real-world graphs often have millions of nodes.
Always sample or filter for visualization.

**Solution**:
```python
# Sample nodes
sampled = random.sample(list(G.nodes()), 200)
subgraph = G.subgraph(sampled)

# Or filter by degree
high_degree = [n for n in G.nodes() if G.degree(n) > 5]
subgraph = G.subgraph(high_degree)
```

#### Challenge 2: Plotly Not Displaying
**Symptom**: Interactive plot doesn't show in Jupyter

**Solutions**:
```bash
# Install/upgrade dependencies
pip install --upgrade plotly jupyter-dash ipywidgets

# Enable extensions
jupyter nbextension enable --py widgetsnbextension
```

#### Challenge 3: GPT-2 Taking Too Long
**Symptom**: Generation is very slow

**Teaching Point**: Large language models are computationally expensive.

**Solutions**:
```python
# Use smaller model
generator = pipeline('text-generation', model='distilgpt2')

# Reduce max length
generator(prompt, max_length=150)  # Instead of 300

# Or use cached results for demos
```

#### Challenge 4: Understanding RAG
**Symptom**: Students don't see why RAG is useful

**Teaching Strategy**:

**Demonstration**:
1. Ask GPT-2 a specific question without RAG
   - Result: Generic or incorrect answer
2. Use RAG with knowledge graph
   - Result: Specific, factually grounded answer
3. Show the difference clearly

**Analogy**:
```
Without RAG: Closed book exam (rely on memory)
With RAG: Open book exam (can reference materials)
```

#### Challenge 5: Poor Query Results
**Symptom**: Retrieval returns "not found" for reasonable queries

**Teaching Point**: Simple string matching has limitations.

**Extension**:
```python
# Fuzzy matching
from fuzzywuzzy import fuzz

def fuzzy_retrieve(query, kg, threshold=80):
    best_match = None
    best_score = 0
    for record in kg:
        score = fuzz.partial_ratio(
            query.lower(), 
            record['subject'].lower()
        )
        if score > best_score and score >= threshold:
            best_score = score
            best_match = record
    return best_match

# Partial matching
def partial_retrieve(query, kg):
    results = []
    for record in kg:
        if query.lower() in record['subject'].lower():
            results.append(record)
    return results[:5]  # Return top 5 matches
```

---

## 📊 Assessment Rubrics

### Lab 1 Rubric (100 points)

| Criterion | Excellent (90-100%) | Good (80-89%) | Adequate (70-79%) | Needs Improvement (<70%) |
|-----------|---------------------|---------------|-------------------|--------------------------|
| **Task 1: Data Loading (10 pts)** | Correctly loads data, handles missing values, drops appropriate columns | Minor issues in missing value analysis | Incomplete missing value handling | Major errors in data loading |
| **Task 2: Text Preprocessing (25 pts)** | Implements all cleaning steps correctly; code is clean and efficient | Missing 1-2 minor steps; code works | Missing major steps or inefficient code | Many preprocessing steps missing |
| **Task 3: Count Vectorizer (15 pts)** | Correct matrix with proper document/feature names | Minor naming issues | Matrix created but incorrectly structured | Major errors in vectorization |
| **Task 4: TF-IDF (20 pts)** | Proper train-test split with stratification; correct vectorization | Minor issues in split or vectorization | Missing stratification or improper fitting | Major errors in splitting or fitting on test |
| **Task 5: Classification (30 pts)** | Trains model, evaluates with multiple metrics, interprets results | Missing 1 metric or interpretation | Model trained but poor evaluation | Model doesn't train or major errors |

### Lab 2 Rubric (100 points)

| Criterion | Excellent (90-100%) | Good (80-89%) | Adequate (70-79%) | Needs Improvement (<70%) |
|-----------|---------------------|---------------|-------------------|--------------------------|
| **Task 1: Graph Creation (20 pts)** | Correctly creates directed graph with all nodes and edges | Minor structural issues | Graph created but missing some elements | Major errors in graph structure |
| **Task 2: Static Visualization (15 pts)** | Clear, well-labeled visualization with appropriate layout | Minor aesthetic issues | Visualization works but unclear | Major display issues |
| **Task 3: Interactive Viz (20 pts)** | Full interactive plot with proper coloring and hover text | Missing minor interactive features | Basic interactivity only | Plot doesn't display or major errors |
| **Task 4: RAG System (45 pts)** | Complete RAG pipeline with retrieval and generation; insightful outputs | Minor issues in retrieval or generation | Basic RAG works but limited functionality | Major errors in implementation |

### Bonus Points (10 points each)

- Implement improved retrieval (fuzzy matching)
- Add multi-hop reasoning
- Create custom visualization styles
- Optimize code for performance
- Add comprehensive error handling
- Write detailed documentation

---

## 💡 Extension Activities

### For Fast Finishers

**Lab 1 Extensions**:

1. **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(tfidf_train_vectors, y_train)
print(f"Best params: {grid_search.best_params_}")
```

2. **Feature Importance Analysis**
```python
# Get top 20 most important words
importances = classifier.feature_importances_
feature_names = tfidf_vectorizer.get_feature_names_out()
top_features = sorted(
    zip(importances, feature_names),
    reverse=True
)[:20]

print("Top 20 Important Words:")
for importance, word in top_features:
    print(f"{word}: {importance:.4f}")
```

3. **Compare Different Models**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC()
}

for name, model in models.items():
    model.fit(tfidf_train_vectors, y_train)
    y_pred = model.predict(tfidf_test_vectors)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}: F1 = {f1:.3f}")
```

**Lab 2 Extensions**:

1. **PageRank Analysis**
```python
# Find most important nodes
pagerank = nx.pagerank(G)
top_nodes = sorted(
    pagerank.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("Top 10 Most Important Entities:")
for node, score in top_nodes:
    print(f"{node}: {score:.4f}")
```

2. **Community Detection**
```python
import networkx.algorithms.community as nx_comm

# Convert to undirected for community detection
G_undirected = G.to_undirected()

# Detect communities
communities = nx_comm.greedy_modularity_communities(G_undirected)

print(f"Found {len(communities)} communities")
for i, community in enumerate(communities[:5]):
    print(f"Community {i}: {len(community)} nodes")
    print(f"Sample members: {list(community)[:5]}")
```

3. **Advanced RAG with Vector Search**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all subjects
subjects = [r['subject'] for r in knowledge_graph]
subject_embeddings = model.encode(subjects)

def semantic_retrieve(query, knowledge_graph, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, subject_embeddings)[0]
    
    # Get top k most similar
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'record': knowledge_graph[idx],
            'similarity': similarities[idx]
        })
    
    return results
```

### Group Activities

**Activity 1: Preprocessing Competition**
- Teams create different preprocessing pipelines
- Compare F1-scores
- Discuss which techniques work best

**Activity 2: Knowledge Graph Design**
- Teams design KG schema for a specific domain
  - Movie database
  - University courses
  - Scientific papers
- Present designs and discuss trade-offs

**Activity 3: RAG Use Case Brainstorm**
- Teams identify real-world RAG applications
- Design system architecture
- Discuss challenges and solutions

---

## 🔍 Discussion Questions

### Lab 1 Discussion

**Conceptual Questions**:
1. Why do we need to preprocess text? What would happen if we didn't?
2. When would high precision be more important than high recall?
3. Why use TF-IDF instead of just counting words?
4. How would you handle class imbalance (e.g., 90% not disaster, 10% disaster)?

**Critical Thinking**:
1. What are the ethical implications of automated disaster detection?
2. How might this system fail with sarcastic tweets?
3. What other features could improve the model (e.g., user profile, time)?
4. How would you deploy this model in production?

### Lab 2 Discussion

**Conceptual Questions**:
1. What are the advantages of knowledge graphs over relational databases?
2. How does RAG reduce hallucination in language models?
3. When would you use RAG vs fine-tuning an LLM?
4. What makes a good knowledge graph schema?

**Critical Thinking**:
1. How would you keep a knowledge graph up-to-date?
2. What are the privacy implications of large knowledge graphs?
3. How can we verify the accuracy of generated answers?
4. What domains would benefit most from RAG systems?

---

## 📝 Sample Solutions

### Lab 1 Task 2 Alternative Solution

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

sw = set(stopwords.words('english'))  # Use set for faster lookup
lemmatizer = WordNetLemmatizer()

def clean_text_v2(text):
    """Alternative cleaning with additional features"""
    
    # Lowercase
    text = text.lower()
    
    # Remove @mentions and #hashtags (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML
    text = re.sub(r'<.*?>', '', text)
    
    # Remove numbers (optional - depends on task)
    # text = re.sub(r'\d+', '', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in sw and len(word) > 2  # Also remove short words
    ]
    
    return ' '.join(tokens)
```

### Lab 2 Task 4 Alternative Solution

```python
def advanced_rag_system(query, knowledge_graph, generator):
    """Enhanced RAG with multiple retrievals and better prompting"""
    
    # Retrieve multiple relevant facts
    results = []
    query_lower = query.lower()
    
    for record in knowledge_graph:
        subject = record['subject'].lower()
        # Check subject and object
        if query_lower in subject or query_lower in record['object'].lower():
            results.append(record)
    
    if not results:
        return f"No information found for '{query}'"
    
    # Format retrieved information
    context = "\n".join([
        f"- {r['subject']} is a {r['predicate']} of type {r['object']}"
        for r in results[:3]  # Top 3 results
    ])
    
    # Enhanced prompt
    prompt = f"""Based on the following facts from a knowledge graph:

{context}

Please answer the question: {query}

Provide a comprehensive answer that synthesizes the above information. If any information is uncertain or missing, please state that clearly.

Answer:"""
    
    # Generate with controlled parameters
    response = generator(
        prompt,
        max_length=300,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )[0]['generated_text']
    
    return response
```

---

## 🎯 Learning Assessment

### Formative Assessment (During Lab)

**Quick Checks**:
- Ask students to explain what each line of code does
- Have students predict output before running code
- Use think-pair-share for conceptual questions
- Monitor notebook progress regularly

**Exit Ticket Questions**:
1. What was the most challenging part today?
2. Name one thing you learned that surprised you.
3. What question do you still have?

### Summative Assessment (End of Lab)

**Practical Skills**:
- Run and evaluate complete notebooks
- Check variable names and outputs match requirements
- Verify understanding through code comments

**Conceptual Understanding**:
- Short quiz on key concepts
- Written explanation of workflow
- Code explanation/walkthrough

**Sample Quiz Questions**:

1. What is the difference between `fit()` and `transform()` in scikit-learn?
2. Why do we stratify when splitting data?
3. In the confusion matrix, what does False Positive mean for disaster tweet classification?
4. What is the main advantage of RAG over standalone LLMs?
5. What does node degree represent in a knowledge graph?

---

## 🛠️ Troubleshooting for Instructors

### Setup Issues

**Issue**: Students have different Python versions
**Solution**: Recommend Python 3.8-3.11, provide Docker image if possible

**Issue**: Package installation fails
**Solution**: 
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Or use conda environment.yml
conda env export > environment.yml
```

### During Session Issues

**Issue**: Code runs on your machine but not students'
**Solution**: Test in clean environment beforehand, have backup VM/cloud notebooks

**Issue**: Students at very different skill levels
**Solution**: 
- Provide "hints" vs "solutions" files
- Pair advanced with beginners
- Prepare extension activities
- Record session for review

**Issue**: Time running short
**Solution**:
- Have "minimum viable" and "full" completion targets
- Skip visualizations if needed (core concepts are more important)
- Assign remaining tasks as homework

---

## 📚 Additional Resources for Instructors

### Slide Decks
Create presentations covering:
- NLP fundamentals (30-40 slides)
- Machine learning basics (20-30 slides)
- Knowledge graphs introduction (20-30 slides)
- RAG concepts (15-20 slides)

### Video Tutorials
Consider recording:
- Environment setup walkthrough
- Each task completion example
- Common error solutions
- Extension activity demonstrations

### Code Templates
Provide:
- Partially completed notebooks with TODOs
- Solution notebooks (hidden initially)
- Common functions library
- Testing/validation scripts

---

## 🎓 Instructor Certification

Instructors should be able to:
- [ ] Complete both labs independently in under 3 hours
- [ ] Explain all concepts without reading notes
- [ ] Debug common errors quickly
- [ ] Adapt content for different skill levels
- [ ] Answer extension questions confidently

---

## 📧 Support for Instructors

For questions or to share improvements:
1. Review the main [README.md](README.md) documentation
2. Check [THEORY.md](THEORY.md) for concept explanations
3. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for rapid lookup
4. Contribute improvements back to the repository

---

**Good luck with your training sessions! 🎉**

*Remember: The goal is not just to complete the tasks, but to understand the concepts and think critically about their applications.*
