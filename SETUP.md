# Setup Checklist

## Pre-Lab Setup (Students)

### ✅ Step 1: Verify Python Installation
```bash
python --version
# Should show Python 3.7 or higher
```

### ✅ Step 2: Create Virtual Environment
```bash
# Navigate to project directory
cd "Gen AI E2 Hands on Lab"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### ✅ Step 3: Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Output**: All packages install successfully without errors.

### ✅ Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Expected Output**: `[nltk_data] Downloaded stopwords...` (success messages)

### ✅ Step 5: Verify Installation
```python
python -c "
import pandas as pd
import numpy as np
import sklearn
import nltk
import networkx as nx
import matplotlib
import seaborn as sns
import plotly
import transformers
print('✅ All packages imported successfully!')
"
```

### ✅ Step 6: Prepare Data Files

**For Lab 1**: 
- [ ] Download `train.csv` from [Kaggle - NLP Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
- [ ] Place in project root directory
- [ ] Verify file size (~3.2 MB, 7,613 rows)

**For Lab 2**:
- [ ] Obtain `dbpedia.csv` file
- [ ] Place in project root directory
- [ ] Verify file has 3 columns: subject, predicate, object

### ✅ Step 7: Launch Jupyter Notebook
```bash
jupyter notebook
# or
jupyter lab
```

**Expected**: Browser opens with Jupyter interface showing notebook files.

### ✅ Step 8: Test Notebook
Open `Tf-idf 1.ipynb` and run the first cell:
```python
import pandas as pd
```

**Expected**: Cell runs without errors.

---

## Troubleshooting Common Issues

### Issue: `command not found: python`
**Solution**: Try `python3` instead of `python`

### Issue: Permission denied during pip install
**Solution**: 
```bash
pip install --user -r requirements.txt
```

### Issue: Virtual environment not activating
**Solution**:
```bash
# Make sure you're in the correct directory
pwd

# Try full path
source /full/path/to/venv/bin/activate
```

### Issue: NLTK download fails
**Solution**:
```python
import nltk
nltk.download('all')  # Downloads all NLTK data (takes longer)
```

### Issue: Jupyter not launching
**Solution**:
```bash
pip install --upgrade jupyter notebook
jupyter notebook --version
```

### Issue: ImportError for transformers
**Solution**:
```bash
pip install --upgrade transformers torch
```

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Python**: 3.7 or higher
- **Internet**: Required for package installation and model downloads

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **Python**: 3.8-3.11
- **Internet**: Stable broadband connection

---

## Pre-Lab Checklist for Instructors

### One Week Before
- [ ] Test all notebooks in clean environment
- [ ] Verify all data files are accessible
- [ ] Prepare slides/presentation materials
- [ ] Send setup instructions to students
- [ ] Create backup cloud notebooks (Colab/Kaggle) if needed

### Day Before
- [ ] Test video conferencing setup (if remote)
- [ ] Prepare Q&A document
- [ ] Review common errors and solutions
- [ ] Test screen sharing
- [ ] Charge devices

### Day Of
- [ ] Arrive 15 minutes early
- [ ] Test internet connection
- [ ] Have backup internet option (mobile hotspot)
- [ ] Open all necessary files
- [ ] Clear terminal history
- [ ] Test audio/video

---

## Alternative Setup Options

### Option 1: Google Colab (No Installation)
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Install packages in first cell:
```python
!pip install nltk networkx plotly transformers
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Option 2: Docker Container
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

Build and run:
```bash
docker build -t genai-lab .
docker run -p 8888:8888 -v $(pwd):/app genai-lab
```

### Option 3: Conda Environment
```bash
conda create -n genai-lab python=3.9
conda activate genai-lab
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Data File Specifications

### train.csv (Lab 1)
- **Source**: Kaggle NLP Getting Started Competition
- **Size**: ~3.2 MB
- **Rows**: 7,613
- **Columns**: 5 (id, keyword, location, text, target)
- **Format**: CSV with header
- **Encoding**: UTF-8

**Sample**:
```csv
id,keyword,location,text,target
1,,,Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all,1
4,,,Forest fire near La Ronge Sask. Canada,1
```

### dbpedia.csv (Lab 2)
- **Source**: DBpedia
- **Format**: CSV without header (3 columns)
- **Columns**: subject, predicate, object
- **Encoding**: UTF-8
- **Expected Rows**: Variable (minimum 50 for basic functionality)

**Sample**:
```csv
Liu Chao-shiuan,Politician,PrimeMinister
Michelle Maylene,Actor,AdultActor
Hirfanlı Dam,Infrastructure,Dam
```

---

## Version Testing Matrix

| Package | Tested Versions | Notes |
|---------|----------------|-------|
| Python | 3.8, 3.9, 3.10, 3.11 | 3.9 recommended |
| pandas | 1.3.0, 1.4.0, 1.5.0 | All compatible |
| scikit-learn | 0.24.0, 1.0.0, 1.1.0 | API stable |
| nltk | 3.6, 3.7, 3.8 | Download data separately |
| transformers | 4.10.0, 4.20.0, 4.30.0 | Newer versions faster |
| plotly | 5.0.0, 5.10.0 | Requires ipywidgets |

---

## Quick Start (TL;DR)

```bash
# 1. Clone/download repository
cd "Gen AI E2 Hands on Lab"

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install packages
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Get data files (train.csv and dbpedia.csv)

# 6. Launch Jupyter
jupyter notebook

# 7. Open notebooks and start learning!
```

---

## Contact & Support

### For Setup Issues
1. Check this document first
2. Review [README.md](README.md) troubleshooting section
3. Search error message online
4. Ask instructor or TA
5. Post in course forum/Discord

### For Content Questions
1. Review [THEORY.md](THEORY.md)
2. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Ask during lab session
4. Office hours

---

## Post-Lab

### After Completing Labs
- [ ] Save final notebooks
- [ ] Export results to PDF
- [ ] Commit to Git (if applicable)
- [ ] Complete feedback survey
- [ ] Backup your work

### Deactivating Environment
```bash
deactivate
```

### Cleaning Up (Optional)
```bash
# Remove virtual environment
rm -rf venv

# Keep notebooks and documentation
```

---

## Success Criteria

You're ready to start when:
✅ Python 3.7+ installed
✅ All packages import successfully
✅ NLTK data downloaded
✅ Data files in place
✅ Jupyter launches
✅ First notebook cell runs

**If all checks pass, you're ready to learn! 🚀**

---

*For issues not covered here, contact your instructor or refer to the main documentation.*
