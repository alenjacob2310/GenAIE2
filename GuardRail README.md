# Guardrails Notebook — README

Overview
- Purpose: A concise developer/architect training reference for the `Guardrails.ipynb` notebook. The notebook demonstrates detecting toxic content and removing sensitive information (PII) using Detoxify, better_profanity, and clean-text.

Dependencies
- Python packages used in the notebook:
  - transformers, detoxify
  - better_profanity
  - clean-text
  - test_guardrails (provided as a wheel for testing integration)

How to run
1. Open `Guardrails.ipynb` in Jupyter or VS Code Notebook.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies (run in PowerShell):
   python -m pip install --upgrade pip
   python -m pip install transformers detoxify better_profanity clean-text
   python -m pip install <path-to>/test_guardrails-0.1-py3-none-any.whl
4. Run cells from top to bottom. The notebook includes example inputs and prints outputs.

Cell-by-cell summary (use cell numbers shown in the notebook UI)
1. Title and project description (markdown).
2. Installs core packages and test wheel (python).
3. Task 1 description: basic toxicity detection (markdown).
4. Installs `detoxify` (python).
5. Implements `detect_toxicity(text)` and tests with a sample string (python).
6. Task 2 description: combine toxicity detection with profanity filter (markdown).
7. Implements `filter_toxic_content(text)` using Detoxify + better_profanity (python).
8. Task 3 description: PII removal (markdown).
9. Implements `detect_pii(text)` using `cleantext.clean(...)` and tests (python).
10. Task 4 description: custom profanity list (markdown).
11. Implements custom bad-words censorship and tests (python).
12. Task 5 description: severity-based profanity censorship (markdown).
13. Implements `censor_severity(text)` and tests (python).
14. Test integration: collects outputs and calls `test_guardrails.save_answer(...)` (python).
15. Empty cell (python).

Expected variables for tests
- `toxicity_detector`, `toxicity_results`, `filtered_output1`, `pii_output`, `filtered_output2`, `censor_output`.

Notes, limitations & tips
- Adjust toxicity threshold and profanity lists for your data.
- PII removal via heuristics may miss cases — use specialized PII detectors for production.
- Model downloads require internet; ensure sufficient disk space.

Troubleshooting
- If installs fail, re-run inside an activated virtual environment.
- For the wheel install, provide an absolute path to the `.whl` file.

Next steps
- Convert sample lists into external config files.
- Add unit tests and CI checks for expected behaviors.
- Replace heuristic PII removal with a dedicated PII detection model.
