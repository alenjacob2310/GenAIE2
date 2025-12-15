# Project Structure and Workflow

## 📁 File Organization

```
Gen AI E2 Hands on Lab/
│
├── 📘 Documentation Files (Read First!)
│   ├── OVERVIEW.md              ⭐ Start here - Navigation guide
│   ├── SETUP.md                 🔧 Installation and setup
│   ├── README.md                📖 Main training documentation
│   ├── QUICK_REFERENCE.md       ⚡ Cheat sheet
│   ├── THEORY.md                🧠 Theoretical concepts
│   ├── INSTRUCTOR_GUIDE.md      👨‍🏫 Teaching resource
│   └── PROJECT_STRUCTURE.md     📊 This file
│
├── 📓 Jupyter Notebooks (Hands-on Labs)
│   ├── Tf-idf 1.ipynb          🐦 Lab 1: Tweet Classification
│   └── Knowledge_graph.ipynb    🕸️ Lab 2: Knowledge Graphs & RAG
│
├── 📦 Configuration Files
│   └── requirements.txt         📋 Python package dependencies
│
└── 📊 Data Files (You provide these)
    ├── train.csv                🚨 Disaster tweets dataset
    └── dbpedia.csv              🌐 Knowledge graph triples
```

---

## 🔄 Learning Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     START YOUR JOURNEY                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   OVERVIEW.md   │
                    │  Choose Path    │
                    └─────────────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │    SETUP.md     │
                    │  Install Tools  │
                    └─────────────────┘
                              │
                              ↓
              ┌───────────────┴───────────────┐
              │                               │
              ↓                               ↓
    ┌─────────────────┐           ┌─────────────────┐
    │   README.md     │           │   THEORY.md     │
    │   Lab 1 Intro   │◄─────────►│  NLP Concepts   │
    └─────────────────┘           └─────────────────┘
              │                               │
              ↓                               │
    ┌─────────────────┐                      │
    │ Tf-idf 1.ipynb  │                      │
    │  Complete Lab   │                      │
    └─────────────────┘                      │
              │                               │
              ↓                               │
    ┌─────────────────┐                      │
    │QUICK_REFERENCE  │◄────(Use During)────┘
    │  Cheat Sheet    │
    └─────────────────┘
              │
              ↓
    ┌─────────────────┐           ┌─────────────────┐
    │   README.md     │           │   THEORY.md     │
    │   Lab 2 Intro   │◄─────────►│ KG & RAG Ideas  │
    └─────────────────┘           └─────────────────┘
              │                               │
              ↓                               │
    ┌─────────────────┐                      │
    │Knowledge_graph  │                      │
    │    .ipynb       │                      │
    │  Complete Lab   │                      │
    └─────────────────┘                      │
              │                               │
              ↓                               │
    ┌─────────────────┐                      │
    │QUICK_REFERENCE  │◄────(Use During)────┘
    │  Cheat Sheet    │
    └─────────────────┘
              │
              ↓
    ┌─────────────────┐
    │   Extensions    │
    │  & Projects     │
    └─────────────────┘
              │
              ↓
    ┌─────────────────┐
    │  COMPLETION! 🎉 │
    └─────────────────┘
```

---

## 📚 Document Dependencies

```
OVERVIEW.md
    │
    ├──► SETUP.md
    │       └──► requirements.txt
    │
    ├──► README.md
    │       ├──► Tf-idf 1.ipynb
    │       │       └──► train.csv
    │       │
    │       └──► Knowledge_graph.ipynb
    │               └──► dbpedia.csv
    │
    ├──► QUICK_REFERENCE.md
    │       ├──► (supports) Tf-idf 1.ipynb
    │       └──► (supports) Knowledge_graph.ipynb
    │
    ├──► THEORY.md
    │       ├──► (explains) Lab 1 concepts
    │       └──► (explains) Lab 2 concepts
    │
    └──► INSTRUCTOR_GUIDE.md
            ├──► (uses) README.md
            ├──► (uses) THEORY.md
            ├──► (uses) QUICK_REFERENCE.md
            └──► (uses) both .ipynb files
```

---

## 🎯 Task Flow - Lab 1

```
┌────────────────────────────────────────────────────────┐
│                    Lab 1: Tweet Classification          │
└────────────────────────────────────────────────────────┘

train.csv  ──────┐
                 │
                 ↓
         ┌───────────────┐
         │   Task 1:     │
         │  Data Loading │
         │   & Cleaning  │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 2:     │
         │     Text      │
         │ Preprocessing │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 3:     │
         │     Count     │
         │  Vectorizer   │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 4:     │
         │    TF-IDF     │
         │ Vectorization │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 5:     │
         │ Random Forest │
         │ Classification│
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │  Evaluation:  │
         │  F1-Score &   │
         │Confusion Matrix│
         └───────────────┘
                 │
                 ↓
            🎯 Results
```

---

## 🕸️ Task Flow - Lab 2

```
┌────────────────────────────────────────────────────────┐
│              Lab 2: Knowledge Graphs & RAG              │
└────────────────────────────────────────────────────────┘

dbpedia.csv ─────┐
                 │
                 ↓
         ┌───────────────┐
         │   Task 1:     │
         │  Load Data &  │
         │  Create Graph │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 2:     │
         │    Static     │
         │ Visualization │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 3:     │
         │  Interactive  │
         │ Visualization │
         └───────────────┘
                 │
                 ↓
         ┌───────────────┐
         │   Task 4:     │
         │      RAG      │
         │ Implementation│
         └───────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ↓                ↓
  ┌─────────────┐  ┌─────────────┐
  │  Retrieval  │  │ Generation  │
  │   Function  │  │   (GPT-2)   │
  └─────────────┘  └─────────────┘
         │                │
         └───────┬────────┘
                 ↓
         ┌───────────────┐
         │   Combined    │
         │   RAG Query   │
         │   Processing  │
         └───────────────┘
                 │
                 ↓
         🎯 Generated Answers
```

---

## 🔍 Code Dependencies

### Lab 1 Package Flow
```
pandas ──┐
numpy ───┤
         ├──► Data Loading & Manipulation
         │
nltk ────┼──► Text Preprocessing
         │    (stopwords, lemmatization)
         │
sklearn ─┴──► Vectorization & Classification
              (TfidfVectorizer, RandomForestClassifier)

matplotlib ──┐
seaborn ─────┤──► Visualization
             │    (confusion matrix)
```

### Lab 2 Package Flow
```
pandas ──────┐
             ├──► Data Loading
networkx ────┤
             ├──► Graph Creation & Analysis
             │
matplotlib ──┤
             ├──► Static Visualization
plotly ──────┤
             ├──► Interactive Visualization
             │
transformers ┴──► Text Generation (GPT-2)
                   for RAG
```

---

## 📊 Data Flow Diagrams

### Lab 1: Data Transformation Pipeline

```
Raw CSV
  │
  ├─ id: 1, 2, 3, ...
  ├─ text: "Forest fire...", "Earthquake...", ...
  ├─ target: 0, 1, 0, 1, ...
  └─ [keyword, location dropped]
  │
  ↓ Task 1: Load & Clean
  │
DataFrame (cleaned)
  │
  ├─ text: "Forest fire...", "Earthquake...", ...
  └─ target: 0, 1, 0, 1, ...
  │
  ↓ Task 2: Preprocess
  │
Cleaned Text
  │
  ├─ text: "forest fire...", "earthquake...", ...
  │   (no URLs, no punctuation, lemmatized)
  │
  ↓ Task 4: Vectorize
  │
TF-IDF Vectors
  │
  ├─ Feature Matrix: (samples × ~12,000 features)
  │   [[0.32, 0.0, 0.15, ...],
  │    [0.0, 0.28, 0.0, ...],
  │    ...]
  │
  ↓ Task 5: Train
  │
Trained Model
  │
  ├─ RandomForestClassifier (100 trees)
  │
  ↓ Predict
  │
Predictions & Metrics
  │
  ├─ Predicted: [0, 1, 1, 0, ...]
  ├─ F1-Score: 0.75
  └─ Confusion Matrix: [[TN, FP], [FN, TP]]
```

### Lab 2: Knowledge Graph to RAG Pipeline

```
CSV Triples
  │
  ├─ (Liu Chao-shiuan, Politician, PrimeMinister)
  ├─ (Michelle Maylene, Actor, AdultActor)
  └─ (Hirfanlı Dam, Infrastructure, Dam)
  │
  ↓ Task 1: Create Graph
  │
NetworkX DiGraph
  │
  ├─ Nodes: 
  │   {Liu Chao-shiuan, PrimeMinister, 
  │    Michelle Maylene, AdultActor, ...}
  │
  ├─ Edges:
  │   [(Liu Chao-shiuan → PrimeMinister, 
  │     relationship="Politician"),
  │    ...]
  │
  ↓ Tasks 2-3: Visualize
  │
Visualizations
  │
  ├─ Static (Matplotlib): Full graph view
  └─ Interactive (Plotly): Filterable subgraph
  │
  ↓ Task 4: Build RAG
  │
Knowledge Graph (first 50)
  │
  └─ List of dictionaries for quick retrieval
  │
  ↓ User Query: "Hohnstein Castle"
  │
Retrieval
  │
  └─ Search result: (Hohnstein Castle, Building, Castle)
  │
  ↓ Combine with prompt
  │
GPT-2 Input
  │
  └─ "Based on: Hohnstein Castle is a Building of type Castle..."
  │
  ↓ Generate
  │
Answer
  │
  └─ "Hohnstein Castle is a historic castle structure..."
```

---

## 🎓 Skill Progression Map

```
Beginner ──────► Intermediate ──────► Advanced
   │                  │                   │
   │                  │                   │
   ↓                  ↓                   ↓
   
Load CSV          Clean Text         Optimize Models
Basic pandas      Regex patterns     Hyperparameter tuning
                  NLTK tools         Cross-validation
   │                  │                   │
   ↓                  ↓                   ↓
   
Simple viz        Vectorization      Advanced NLP
matplotlib        TF-IDF             BERT, GPT
bar charts        Train-test split   Transfer learning
                  
   │                  │                   │
   ↓                  ↓                   ↓
   
Read graphs       Build graphs       Analyze graphs
NetworkX basics   Add nodes/edges    PageRank, Communities
                  Layouts            Complex queries
   │                  │                   │
   ↓                  ↓                   ↓
   
Use models        Evaluate models    Build systems
predictions       Metrics            RAG pipelines
                  Confusion matrix   Production deployment
```

---

## 🔄 Iterative Learning Cycle

```
        ┌────────────────┐
        │   1. Read      │
        │  Documentation │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │  2. Understand │
        │    Concepts    │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │  3. Practice   │
        │   in Notebook  │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │   4. Debug     │
        │   & Refine     │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │  5. Review     │
        │   Theory       │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │  6. Extend     │
        │   & Apply      │
        └────────┬───────┘
                 │
                 ↓
        ┌────────────────┐
        │  7. Master! 🎓 │
        └────────────────┘
```

---

## 📈 Time Investment Breakdown

```
Activity                    Time        Cumulative
─────────────────────────────────────────────────
Setup & Installation        30-45 min   0:45
Read OVERVIEW & SETUP       15 min      1:00
Read README Lab 1           30 min      1:30
Complete Lab 1 Tasks        90-120 min  3:30
Read THEORY (NLP)           30 min      4:00
─────────────────────────────────────────────────
Break / Review              30 min      4:30
─────────────────────────────────────────────────
Read README Lab 2           20 min      4:50
Complete Lab 2 Tasks        60-90 min   6:20
Read THEORY (KG & RAG)      30 min      6:50
─────────────────────────────────────────────────
Extensions (optional)       60-120 min  8:50
Review & Consolidation      30 min      9:20
─────────────────────────────────────────────────
Total                       ~8-10 hours
```

---

## 🎯 Completion Checklist

### Documentation Review
- [ ] Read OVERVIEW.md
- [ ] Completed SETUP.md
- [ ] Studied relevant sections of README.md
- [ ] Referenced QUICK_REFERENCE.md during labs
- [ ] Read THEORY.md for concepts
- [ ] (Instructors) Reviewed INSTRUCTOR_GUIDE.md

### Lab 1 Completion
- [ ] Task 1: Data loading ✓
- [ ] Task 2: Text preprocessing ✓
- [ ] Task 3: Count Vectorizer ✓
- [ ] Task 4: TF-IDF ✓
- [ ] Task 5: Classification ✓
- [ ] F1-Score > 0.70 ✓

### Lab 2 Completion
- [ ] Task 1: Graph creation ✓
- [ ] Task 2: Static visualization ✓
- [ ] Task 3: Interactive visualization ✓
- [ ] Task 4: RAG implementation ✓
- [ ] Query returns meaningful answers ✓

### Understanding Check
- [ ] Can explain TF-IDF vs BoW
- [ ] Understand precision vs recall tradeoff
- [ ] Know when to use stratified splitting
- [ ] Can describe knowledge graph structure
- [ ] Understand RAG benefits over plain LLMs

---

## 🚀 Where to Go From Here

```
Current State: Completed Labs
         │
         ↓
    Choose Path:
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
    ↓         ↓        ↓        ↓
 Personal  Advanced  Career   Research
 Projects  Courses   Track    & Papers
    │         │        │        │
    ↓         ↓        ↓        ↓
 Portfolio  BERT/     Job      Latest
 Building   GPT      Search   Methods
            │         │        │
            └────┬────┴────┬───┘
                 │         │
                 ↓         ↓
          ML Engineering  NLP Research
           (Production)   (Innovation)
```

---

## 💡 Pro Tips for Navigation

1. **Bookmark this file** - Quick reference to project structure
2. **Use Ctrl+F** - Search for specific topics
3. **Follow the workflow** - Don't skip steps
4. **Print QUICK_REFERENCE.md** - Keep handy during coding
5. **Revisit THEORY.md** - Deepen understanding over time

---

## 📞 Quick Help Guide

**Issue**: Can't find information
**Solution**: Check this structure, use document index in OVERVIEW.md

**Issue**: Don't understand a concept
**Solution**: THEORY.md → README.md example → Try in notebook

**Issue**: Code not working
**Solution**: QUICK_REFERENCE.md → README.md troubleshooting → SETUP.md

**Issue**: Want to teach this
**Solution**: INSTRUCTOR_GUIDE.md → Practice labs yourself → Prepare materials

---

**You now have a complete map of the project! Choose your starting point and begin learning! 🎯**

*Refer back to this document whenever you feel lost or need to understand how pieces fit together.*
