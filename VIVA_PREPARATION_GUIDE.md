# TextGuard AI — Project Presentation & Viva Preparation Guide

**American International University-Bangladesh (AIUB)**  
**Course:** CSC 4162 — Programming in Python  
**Project:** TextGuard AI — AI Text Detection System Using Python & Machine Learning  

---

## 1. Project Summary & Demonstration Strategy

### Demonstration Sequence for Viva:
1. **Introduction:** Open `main.py` in the terminal. Point out the modular file structure (`gui/`, `ml/`, `analysis/`, `data/`, `models/`, `tests/`).
2. **Launch Application:** Run `python main.py`. Demonstrate the modern dark-themed CustomTkinter desktop interface.
3. **Live Human Sample Test:** Click "Load Human Sample" in the editor. Point out the live token counters updating in real time (84 words, 486 characters, 4 sentences).
4. **Trigger Analysis:** Click "Analyze Text" (or press `Ctrl+Enter`). Note that the GUI remains smooth and responsive due to background `threading.Thread`.
5. **Explain Verdict & Scores:** Show the green "More Likely Human-Written" badge (96.2% Human score, 3.8% AI score). Explain that the scores are calibrated probabilities from Logistic Regression.
6. **Explain Stylometric Metrics:** Show the metric cards:
   - Lexical Diversity (TTR = 0.824, high variety)
   - Burstiness (8.42 std dev, human sentence rhythm variation)
   - Avg Sentence Length (21.0 words)
   - Readability (Flesch score 72.4)
7. **Show Embedded Visualizations:** Point to the probability gauge bar chart and the sentence rhythm progression line plot.
8. **Live AI Sample Test:** Click "Load AI Sample" and analyze. Show the red "More Likely AI-Generated" badge (94.8% AI score), lower burstiness, and detected transition phrases ("furthermore", "in conclusion").
9. **Export Feature:** Click "PDF Report" and show the generated professional PDF report in Adobe Acrobat / browser.
10. **History & Search:** Click "Analysis History". Show search filtering, inspecting past records, and privacy settings.
11. **Run Unit Tests:** In terminal, execute `pytest -v` to prove all 25 unit tests pass cleanly.

---

## 2. In-Depth Technical Viva Questions & Answers

### Python Language Concepts
**Q1: How does your project demonstrate Object-Oriented Programming (OOP)?**  
*Answer:* We designed 10 specialized classes across the project. For example:
- `TextGuardApp` inherits from `ctk.CTk` and acts as the master coordinator.
- `InputPanel`, `ResultsPanel`, `ChartsPanel`, and `HistoryPanel` inherit from `ctk.CTkFrame`, demonstrating encapsulation of GUI components.
- `LinguisticAnalyzer` encapsulates stylometric feature extraction with private methods like `_count_syllables_word()`.
- `HistoryManager` encapsulates file operations and CRUD state for analysis records.

**Q2: What data structures did you use and why?**  
*Answer:*
- **List:** Used to store sentences, token sequences, and chronological history records where order matters.
- **Dictionary:** Used for prediction results, linguistic metric payloads, and theme color definitions for fast key-value lookup and direct JSON serialization.
- **Set:** Used for stopword sets and transition words to achieve $O(1)$ constant-time membership lookups during text parsing.
- **Tuple:** Used for immutable coordinate pairs, RGB values, and word-frequency pairs `(word, count)`.

**Q3: Where and how did you implement Exception Handling?**  
*Answer:* We placed targeted `try-except` blocks around external dependencies and file operations:
- In `ml/predictor.py`: Catches missing or corrupted model files (`FileNotFoundError`, `RuntimeError`) and displays an offline status rather than crashing.
- In `gui/input_panel.py`: Catches file read errors with `errors='replace'` to handle non-UTF-8 encodings.
- In `analysis/report_generator.py`: Safely checks whether ReportLab is installed before attempting PDF exports.
- In `data/history_manager.py`: Automatically initializes a valid JSON document if the history file is empty or corrupted.

---

### Machine Learning & NLP Concepts
**Q4: Explain the dataset you used for training.**  
*Answer:* We used the `AI_vs_Human_Text_Classification_Benchmark.csv` dataset containing 50,000 samples. It is strictly balanced with 25,000 human-written samples and 25,000 AI-generated samples from 5 major models: ChatGPT-4o, Claude 3.5 Sonnet, Llama 3 70B, Gemini 1.5 Pro, and Mistral Large.

**Q5: What is TF-IDF and why did you use word n-grams (1, 2)?**  
*Answer:* Term Frequency-Inverse Document Frequency (TF-IDF) converts text strings into numerical vectors by weighing words that are frequent in a specific passage against their frequency across the entire corpus. Using both unigrams (single words) and bigrams (pairs of words, e.g., "contemporary discourse", "in conclusion") allows the model to capture recurring phrasing patterns characteristic of synthetic AI language models.

**Q6: What is the difference between a raw classification margin and a calibrated probability?**  
*Answer:* In linear models like SVM, the raw decision function outputs an unconstrained distance from the separating hyperplane, which cannot be directly interpreted as a percentage. In contrast, Logistic Regression applies the sigmoid function $P(y=1) = \frac{1}{1 + e^{-z}}$, producing mathematically sound, calibrated posterior probabilities between 0.0 and 1.0.

**Q7: How does your system handle inconclusive cases?**  
*Answer:* Rather than forcing a binary classification when the model is uncertain, `TextGuardPredictor` checks the margin between AI and Human probabilities. If the probability falls within the uncertainty band ($42.5\% \le P \le 57.5\%$), the system flags the text as **"Inconclusive / Borderline"** with a "Low / Uncertain" confidence rating, preventing false accusations.

---

### Concurrency & Architecture
**Q8: Why is threading important in a desktop GUI application?**  
*Answer:* Desktop GUI event loops (like Tkinter's `mainloop()`) run on the main thread. If intensive computation (like vectorizing long passages or calculating linguistic variance) runs synchronously on the main thread, the window freezes, stops responding to clicks, and triggers an OS "Not Responding" error. By executing `run_async_analysis()` in a daemon `threading.Thread` and returning results via `self.after(0, callback)`, the GUI remains 100% smooth and interactive.

**Q9: How do your unit tests verify software quality?**  
*Answer:* We implemented 25 comprehensive unit tests using `pytest` covering:
- Input validation (empty text, sub-minimum words, character boundaries).
- Stylometric correctness (TTR calculations, sentence std dev, transition detection).
- Model inference reliability on human, AI, and borderline texts.
- Report export generation (verifying TXT, JSON, and PDF files).
- History CRUD operations and privacy controls.
