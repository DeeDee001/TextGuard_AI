# Data Directory — TextGuard AI

This directory manages test samples, datasets, and persistent analysis history for the TextGuard AI system.

## Files Description

1. **`AI_vs_Human_Text_Classification_Benchmark.csv`**:
   - Primary benchmark dataset used for model training and evaluation.
   - **Records:** 50,000 samples (25,000 Human-written, 25,000 AI-generated).
   - **AI Generation Models Represented:** ChatGPT-4o, Claude 3.5 Sonnet, Llama 3 70B, Gemini 1.5 Pro, and Mistral Large.
   - **Labels:** `0` = Human Origin, `1` = AI Generated.
   - **License:** Open Research Dataset for academic text classification.

2. **`history.json`**:
   - Persistent JSON store for user analysis sessions (satisfies AIUB Project Requirement FR-3).
   - Stores analysis timestamp, text preview, word count, character count, model verdict, probability percentages, and stylometric metrics.
   - User privacy controls can disable history recording or redact full text content.

3. **`sample_human_text.txt`**:
   - Natural human writing sample covering university experiences, reflective thoughts, and programming reflections.

4. **`sample_ai_text.txt`**:
   - Synthetic AI writing sample featuring formal discourse markers ("contemporary discourse", "stakeholders", "furthermore", "in conclusion").
