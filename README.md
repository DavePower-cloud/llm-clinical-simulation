# 🧠 LLM Clinical Simulation

Multi-agent large language model framework for simulating clinical conversations in healthcare training environments, with automated evaluation using LLM-based judging.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)
![LLMs](https://img.shields.io/badge/LLMs-Generative%20AI-purple)  
![Multi-Agent Systems](https://img.shields.io/badge/Multi--Agent%20Systems-Agentic%20AI-blue)  
![AI Domain](https://img.shields.io/badge/AI-Healthcare%20Simulation-green)  


---

## 📄 Associated Research

This repository accompanies the following research work:

- **Power D & Power T.** *Can large language models simulate the clinical conversation? A multi-agent approach?* (Preprint available) https://doi.org/10.35542/osf.io/etv6d_v1

---

## 📌 Overview

Clinical communication is central to safe patient care, yet studying communication behaviour is difficult due to the cost, complexity, and ethical constraints of real-world observation and simulation.

This project investigates whether large language models (LLMs) can simulate realistic clinical conversations between healthcare professionals and patients.

The framework uses multiple LLM agents assigned specific clinical roles (e.g., doctor, nurse, patient). These agents interact within a controlled conversation loop, producing simulated dialogues representing clinical scenarios.

Generated conversations are then evaluated automatically using a LLM-based judge model, enabling scalable, reproducible assessment of communication quality.

---

## ⚙️ Key Features

- Multi-agent LLM clinical conversation simulation  
- Role-locked agents (doctor, nurse, patient)  
- Support for multiple providers (OpenAI, Anthropic, Google)  
- Structured prompt templates for each role  
- Conversation persistence in JSON / JSONL format  
- Batch generation of clinical scenarios  
- Automated evaluation using AI-as-judge  
- Designed for reproducible research workflows  

---

## 🏗️ Repository Structure

src/
- simulation.py # Core multi-agent simulation engine
- clients.py # LLM provider wrappers
- prompts.py # Role-specific system prompts
- models.py # Data structures
- judge.py # AI-as-judge scoring logic
- evaluation.py # Batch evaluation pipeline
- config.py # Configuration management
- io_utils.py # File handling utilities
- main.py # CLI entry point

notebooks/
- demo_generation.ipynb
- as_as_judge.ipynb

data/
- conversations/ # Generated conversations
- judge_results.jsonl

prompts/
- doctor_prompt.md
- patient_prompt.md
- nurse_prompt.md

---

## 🚀 How to Run

### 1. Clone the repository
git clone https://github.com/DavePower-cloud/llm-clinical-simulation.git
cd llm-clinical-simulation

### 2. Install dependencies
pip install -r requirements.txt

### 3. Configure API keys
Create a .env file in the root directory:

OPENAI_API_KEY=your_key_here \
ANTHROPIC_API_KEY=your_key_here \
GOOGLE_API_KEY=your_key_here

### 4. Run a single simulation
python -m src.main --verbose

Output will be saved to:

data/conversations/

### 5. Run batch simulations
python -m src.main --batch 10

### 6. Run AI-based evaluation
python -m src.evaluation \
  --input-dir data/conversations \
  --output-jsonl data/judge_results.jsonl

Optional per-file outputs:

python -m src.evaluation \
  --input-dir data/conversations \
  --output-jsonl data/judge_results.jsonl \
  --per-file-output-dir data/judge_results

### ⚡ Quick Demo (End-to-End)

Run a full pipeline: \
python -m src.main --batch 3 \
python -m src.evaluation --input-dir data/conversations

---

### 📊 Output Formats
Conversation Output (JSON)

Each simulation produces a structured JSON file:

turn-by-turn dialogue

speaker roles

timestamps

metadata (e.g., role guard failures)


### 📄 Example Conversation Output


{\
  "conversation_id": "conv_1234abcd", \
  "num_turns": 6, \
  "role_guard_failures": 0, \
  "turns": [ \
    {"speaker": "doctor", "text": "Can you describe your chest pain?"}, \
    {"speaker": "patient", "text": "It's crushing and spreading to my arm."}, \
    {"speaker": "nurse", "text": "BP is 88/54, HR 120."} \
  ] \
} 



Evaluation Output (JSONL)

Each evaluation includes:

Likert scores:

role fidelity

coherence

communication realism

educational usability

free-text justification

model metadata

### 📊 Example Evaluation Output

{\
  "conversation_id": "conv_1234abcd", \
  "judge_model": "gpt-4o", \
  "judge": { \
    "role_fidelity": 5, \
    "turn_coherence": 4, \
    "communication_realism": 5, \
    "educational_usable": true, \
    "comments": "Clinically plausible interaction with appropriate escalation." \
  } \
}

---

## 🔬 Research Applications

This framework supports:

Simulation-based medical education research

Evaluation of structured communication protocols (e.g., ISBAR)

Synthetic dataset generation for training and benchmarking

AI-assisted scenario design and assessment

Large-scale analysis of clinical communication behaviour

---

## 📦 Dataset & Future Work

This repository forms part of the ESCALATE dataset pipeline, which will include:

Large-scale synthetic clinical conversation datasets

Structured JSONL releases

Human and AI evaluation benchmarks

Public release via Zenodo and Hugging Face

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.

It is not intended for clinical use or real-world decision-making.

---

## 👤 Author

**David Power**  
Healthcare Simulation Specialist | MSc Artificial Intelligence  

- 💼 LinkedIn: https://www.linkedin.com/in/dave-power-47280a44/  
- 💻 GitHub: https://github.com/DavePower-cloud

---

## 📜 License

MIT License

---

## 📚 Citation

If you use this work, please cite the associated publications.

---

## 🔁 Pipeline Overview

```mermaid
flowchart LR
    A[Scenario Definition] --> B[Multi-Agent Simulation]
    B --> C[Conversation JSON Output]
    C --> D[AI Judge Evaluation]
    D --> E[Structured Scores JSONL]
    E --> F[Dataset / Analysis / Publication]



