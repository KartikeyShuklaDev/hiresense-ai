# 🚀 HireSense AI  
### AI‑Powered Voice Interview Platform with RAG, Multilingual Support & Flutter UI

**Author:** Kartikey Shukla  
**GitHub:** https://github.com/KartikeyShuklaDev  

---

## 📌 Overview

HireSense AI is an end‑to‑end **AI‑driven recruitment and interview automation platform** that simulates a real human interviewer using **voice interaction**, **Retrieval‑Augmented Generation (RAG)**, and **Large Language Models (LLMs)**.

The platform automates resume understanding, HR and technical interviews, candidate evaluation, and recruitment communication (MailGen). It is built with a **Flutter frontend** and a **Flask‑based AI backend**, making it suitable for real‑world hiring, campus recruitment, and research use cases.

---

## ✨ Key Features

- 🎙️ Fully voice‑based AI interviewer  
- 🌍 Multilingual interview support  
- 🧠 RAG‑powered technical question generation  
- 📱 Cross‑platform Flutter UI (Web & Mobile)  
- 📩 Automated recruitment emails (MailGen)  
- 🔐 Secure, modular, production‑ready architecture  

---

## 🗣️ Voice‑Based AI Interviewer

HireSense AI conducts interviews through **natural voice interaction**, eliminating manual input and simulating a real interview experience.

**Highlights**
- No keyboard or mouse required  
- Human‑like conversational flow  
- Automatic voice detection  
- Graceful handling of pauses and retries  
- Interviewer persona: *Victus*  

---

## 🌍 Multilingual Support

Candidates can choose their preferred interview language at the beginning of the session:

- English  
- Hindi  
- Punjabi  
- Marathi  
- Tamil  

Speech‑to‑Text (STT) and Text‑to‑Speech (TTS) dynamically adapt to the selected language.

---

## 🧠 Intelligent Interview Workflow

1. **Candidate Onboarding**  
   - Candidate name and language selection  
   - Secure data storage  

2. **HR Interview Round**  
   - Structured and bias‑aware HR questions  
   - Focus on communication and behavioral skills  

3. **Technical Interview Round**  
   - Candidate selects technical skills (e.g., Python, Java, C++)  
   - Skill‑conditioned questions generated using RAG  
   - Questions grounded strictly in textbook‑verified content  

4. **Wrap‑Up**  
   - Candidate queries  
   - Voice‑based interview summary  

---

## 📚 Retrieval‑Augmented Generation (RAG)

HireSense AI uses a semantic **RAG pipeline** to ensure accurate, grounded, and hallucination‑free technical interviews.

**RAG Pipeline**
- Textbooks & PDFs are chunked and embedded  
- FAISS vector database enables semantic retrieval  
- Skill‑conditioned semantic queries are generated  
- LLM produces questions strictly from retrieved context  

**Benefits**
- Domain‑accurate interviews  
- No hallucinated questions  
- Transparent and explainable evaluation  

---

## 📊 Evaluation & Metrics

### Candidate Evaluation
- Accuracy score (0–100)  
- Missing concept detection  
- Groundedness with retrieved context  
- LLM confidence‑based reliability  

### RAG Evaluation
- Precision  
- Recall  
- F1‑Score  
- Concept‑level relevance analysis  

Metrics are aggregated at the interview level and visualized for analysis.

---

## 📱 Flutter UI (Frontend)

HireSense AI uses **Flutter** to provide a modern, cross‑platform user interface.

**Why Flutter**
- Single codebase for Web & Mobile  
- High‑performance UI  
- Real‑time voice interaction  
- Clean separation from backend AI logic  

**Flutter Responsibilities**
- Candidate onboarding screens  
- Voice interview interface  
- Interview progress tracking  
- Status and history views  
- Display of evaluation summaries  

Flutter handles only presentation; all AI logic runs on the backend.

---

## 🧩 Technology Stack

**Frontend**
- Flutter (Web & Mobile)

**Backend & AI**
- Python (Flask microservices)  
- FAISS (Vector Database)  
- MongoDB (Candidate & interview data)  

**LLMs & Speech**
- Groq LLM – LLaMA‑4 Maverick (primary reasoning)  
- Gemini (fallback LLM & STT)  
- ElevenLabs (STT & TTS)  
- Whisper (fallback STT)  
- Sentence Transformers (embeddings)  

---

## 📁 Project Structure

```text
integration-with-flutter/
│
├── frontend_app/                         # Flutter Frontend (Web / Mobile)
│   ├── lib/
│   │   ├── screens/                      # UI Screens
│   │   │   ├── home_screen.dart
│   │   │   ├── interview_screen.dart
│   │   │   ├── status_screen.dart
│   │   │   └── history_screen.dart
│   │   │
│   │   ├── services/                     # API Communication Layer
│   │   │   ├── api_client.dart
│   │   │   └── interview_service.dart
│   │   │
│   │   ├── widgets/                      # Reusable UI Components
│   │   │   ├── animated_wave.dart
│   │   │   ├── start_button.dart
│   │   │   ├── loading_indicator.dart
│   │   │   └── status_card.dart
│   │   │
│   │   ├── utils/                        # Helper Functions
│   │   ├── constants/                    # App Constants
│   │   └── main.dart                     # Flutter Entry Point
│   │
│   ├── android/                          # Android Platform Files
│   ├── ios/                              # iOS Platform Files
│   ├── web/                              # Web Build
│   ├── windows/ linux/ macos/            # Desktop Platforms
│   ├── pubspec.yaml                      # Flutter Dependencies
│   └── README.md
│
├── HRInterviewer/                        # AI Interview Backend
│   └── backend/
│       ├── app.py                        # Flask Application Entry
│       ├── config.py                     # Environment & Config
│       │
│       ├── routes/                       # API Routes
│       │   ├── interview.py
│       │   └── speech.py
│       │
│       ├── services/                     # Core AI Logic
│       │   ├── interview_service.py
│       │   ├── rag_service.py
│       │   ├── rag_metrics.py
│       │   ├── speech_service.py
│       │   ├── elevenlabs_service.py
│       │   └── gemini_service.py
│       │
│       ├── vector_db/                    # FAISS Vector Database
│       │
│       ├── data/
│       │   ├── books/                    # Textbooks / PDFs (RAG)
│       │   ├── audio/                    # Temporary Audio Files
│       │   └── hr_questions.json
│       │
│       ├── utils/                        # Audio & Helper Utilities
│       ├── requirements.txt              # Python Dependencies
│       └── start_backend.bat
│
├── .gitignore
├── LICENSE
└── README.md

