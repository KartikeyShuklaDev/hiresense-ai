🚀 HireSense AI
AI‑Powered Voice Interview Platform with RAG, Multilingual Support & Real‑Time Evaluation
Author: Kartikey Shukla
GitHub: https://github.com/KartikeyShuklaDev

📌 Overview
HireSense AI is an end‑to‑end AI‑driven recruitment and interview automation platform that simulates a real human interviewer using voice interaction, Retrieval‑Augmented Generation (RAG), and Large Language Models (LLMs).

The platform automates:

Resume understanding

HR and technical interviews

Candidate evaluation

Recruitment communication (MailGen)

HireSense AI is designed for scalable hiring, campus recruitment, mock interviews, and research on RAG evaluation, with a Flutter frontend and a Flask‑based AI backend.

✨ Key Features
🎙️ Fully voice‑based AI interviewer

🌍 Multilingual interview support

🧠 RAG‑powered technical question generation

📊 Concept‑level Precision, Recall, and F1‑Score evaluation

📱 Flutter UI for Web & Mobile

📩 Automated recruitment emails (MailGen)

🔐 Secure, modular, production‑ready architecture

🗣️ Voice‑Based AI Interviewer
HireSense AI conducts interviews using natural voice interaction, eliminating manual input.

Highlights
No keyboard or mouse required

Human‑like conversational flow

Automatic voice detection

Graceful handling of pauses and retries

Interviewer persona: Victus

🌍 Multilingual Support
Candidates can select their preferred language at the start of the interview:

English

Hindi

Punjabi

Marathi

Tamil

Speech‑to‑Text (STT) and Text‑to‑Speech (TTS) dynamically adapt based on the selected language.

🧠 Intelligent Interview Flow
1️⃣ Candidate Onboarding
Candidate name & language selection

Secure data storage in MongoDB

2️⃣ HR Interview Round
Structured, bias‑aware HR questions

Focus on communication, behavior, and situational awareness

No salary‑related questions in early stages

3️⃣ Technical Interview Round
Candidate selects technical skills (e.g., Python, Java, C++)

Skill‑conditioned questions generated using RAG

Questions grounded strictly in textbook‑verified content

4️⃣ Wrap‑Up
Candidate questions

Voice‑based interview summary

📚 Retrieval‑Augmented Generation (RAG)
HireSense AI uses a semantic RAG pipeline to ensure accurate, grounded, and hallucination‑free interviews.

RAG Pipeline
📖 Textbooks & PDFs → chunked and embedded

🔍 FAISS vector database for semantic retrieval

🧩 Skill‑conditioned semantic queries

🧠 LLM generates questions only from retrieved context

Benefits
No hallucinated questions

Domain‑accurate technical interviews

Transparent and explainable evaluation

📊 Evaluation & Metrics
Candidate Evaluation
Accuracy score (0–100)

Missing concept detection

Groundedness with retrieved context

LLM confidence‑based reliability score

RAG Evaluation
Precision

Recall

F1‑Score

Concept‑level relevance analysis

Aggregated interview‑level metrics

Metrics are visualized using clear bar charts for analysis and reporting.

📱 Flutter UI (Frontend)
HireSense AI uses Flutter to deliver a cross‑platform, real‑time voice interview interface.

Why Flutter?
Single codebase for Web & Mobile

High‑performance UI

Real‑time voice interaction

Clean separation from AI logic

Easy REST API integration

Flutter Responsibilities
Candidate onboarding & interview screens

Voice input/output handling

Interview progress tracking

Status and history views

Display of evaluation summaries

Flutter handles presentation only; all AI logic runs on the backend.

🧩 Technology Stack
Frontend
Flutter (Web & Mobile)

Backend & AI
Python (Flask microservices)

FAISS (Vector Database)

MongoDB (Candidate & interview data)

LLMs & Speech
Groq LLM – LLaMA‑4 Maverick (primary reasoning & evaluation)

Gemini (fallback LLM & STT)

ElevenLabs (STT & TTS)

Whisper (fallback STT)

Sentence Transformers (embeddings)

📁 Project Structure
integration-with-flutter/
│
├── frontend_app/                  # Flutter Frontend (Web / Mobile)
│   ├── lib/
│   │   ├── screens/               # UI Screens
│   │   ├── services/              # API Communication
│   │   ├── widgets/               # Reusable UI Components
│   │   └── main.dart
│   │
│   └── pubspec.yaml
│
├── HRInterviewer/
│   └── backend/
│       ├── app.py                 # Flask App Entry
│       ├── routes/                # REST APIs
│       ├── services/              # RAG, LLM, Speech, Evaluation
│       ├── vector_db/              # FAISS Index
│       ├── data/                   # Textbooks & Audio
│       └── requirements.txt
│
├── .gitignore
├── LICENSE
└── README.md
🔁 End‑to‑End System Flow
Flutter UI → Flask APIs → RAG + LLM → Evaluation → Results → Flutter UI → MailGen

📈 Use Cases
AI‑driven technical hiring

Campus recruitment automation

Mock interview practice

Skill‑based candidate screening

Research on RAG evaluation metrics

🔐 Security & Best Practices
No API keys committed

.env‑based configuration

Clean Git history

Modular and scalable design

Production‑ready repository structure

👨‍💻 About the Author
Kartikey Shukla
AI & Full‑Stack Developer
Focused on LLMs, RAG systems, Voice AI, Flutter applications, and intelligent evaluation frameworks.

🔗 GitHub: https://github.com/KartikeyShuklaDev

