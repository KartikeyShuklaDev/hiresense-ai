
import sys
import io
import wave
import os
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Optional heavy native packages (may not be available on all systems,
# especially CI or minimal Windows installs). Import lazily and fall back
# so the backend can start without audio support.
try:
    # use importlib to avoid static analysis errors when numpy is not installed
    import importlib
    np = importlib.import_module("numpy")
except Exception:
    np = None

try:
    import importlib
    sd = importlib.import_module("sounddevice")
except Exception:
    sd = None

try:
    import importlib
    _gr_module = importlib.import_module("groq")
    # Prefer the Groq class if provided, otherwise keep the module (best-effort)
    Groq = getattr(_gr_module, "Groq", _gr_module)
except Exception:
    Groq = None

try:
    import importlib
    # Dynamically import google.genai to avoid static analysis/import-time errors
    try:
        genai = importlib.import_module("google.genai")
    except Exception:
        # Some environments may expose genai as an attribute on the google package
        try:
            google_pkg = importlib.import_module("google")
            genai = getattr(google_pkg, "genai", None)
        except Exception:
            genai = None

    # Attempt to get the types submodule/attribute if available
    ga_types = None
    if genai is not None:
        ga_types = getattr(genai, "types", None)
        if ga_types is None:
            try:
                ga_types = importlib.import_module("google.genai.types")
            except Exception:
                ga_types = None
except Exception:
    genai = None
    ga_types = None
try:
    import importlib
    pyttsx3 = importlib.import_module("pyttsx3")  # local TTS engine (lazy import to avoid static analysis errors)
except Exception:
    pyttsx3 = None

# Lazy-import pymongo to avoid static-analysis/import-time errors when pymongo is not installed.
try:
    import importlib as _importlib
    _pymongo = _importlib.import_module("pymongo")
    MongoClient = getattr(_pymongo, "MongoClient", None)
except Exception:
    _pymongo = None
    MongoClient = None
    print("⚠ pymongo not available; MongoDB features will be disabled.")

# Make backend root importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

print("✅ Using backend root:", ROOT_DIR)

# ---- IMPORT EXISTING INTERVIEW LOGIC (RAG + Llama 4 Maverick) ----
# Note: heavy ML/RAG imports are performed lazily inside functions to allow
# the Flask app to start even if optional model libraries are not installed.

# ---- ELEVENLABS SERVICE (TTS + STT) ----
# Lazy import elevenlabs service (may depend on pydantic_core etc.). If
# it fails we'll provide no-op fallbacks so the server can still start.
try:
    from services.elevenlabs_service import eleven_tts, eleven_stt
except Exception as _e:
    print(f"⚠ elevenlabs_service import failed: {_e}")
    def eleven_tts(text: str) -> bool:
        print("⚠ eleven_tts not available (fallback).")
        return False

    def eleven_stt(wav_bytes: bytes) -> str:
        print("⚠ eleven_stt not available (fallback). Returning empty transcription.")
        return ""

# ========================= API KEYS & CLIENTS =========================

# 🔥 Put your real keys here or via env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Create API clients only if the libraries are available; otherwise leave None
client_gemini = None
client_groq = None
if genai is not None and GEMINI_API_KEY:
    try:
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as _e:
        print(f"⚠ Could not initialize Gemini client: {_e}")

if Groq is not None and GROQ_API_KEY:
    try:
        client_groq = Groq(api_key=GROQ_API_KEY)
    except Exception as _e:
        print(f"⚠ Could not initialize Groq client: {_e}")

# Local TTS engine (no quota)
engine_tts = None
if pyttsx3 is not None:
    try:
        engine_tts = pyttsx3.init()
    except Exception:
        engine_tts = None

# ========================= MONGODB SETUP =========================

try:
    mongo_client = MongoClient(
        "mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=2000
    )
    mongo_client.admin.command("ping")
    mongo_db = mongo_client["HireSense"]
    mongo_candidates = mongo_db["CandidateData"]
    mongo_sessions = mongo_db["InterviewSessions"]
    MONGO_OK = True
    print("✅ Connected to MongoDB (HireSense.CandidateData).")
except Exception as e:
    print(f"⚠ MongoDB not available: {e}")
    mongo_client = None
    mongo_db = None
    mongo_candidates = None
    MONGO_OK = False
    mongo_sessions = None

# ========================= AUDIO SETTINGS =========================

SAMPLE_RATE = 24000
CHANNELS = 1
ANSWER_RECORD_SECONDS = 25  # time for main answers

# Flags to disable cloud TTS once they fail
GEMINI_TTS_ENABLED = True
GROQ_TTS_ENABLED = True

# HR questions file
HR_QUESTIONS_PATH = ROOT_DIR / "data" / "hr_questions.json"

# How many non-pass answers are required
HR_REQUIRED = 3
TECH_REQUIRED = 3

# ========================= LIVE STATUS & SESSION =========================
# Exposed via InterviewController.get_status() for Flutter polling
current_status = {
    "name": None,
    "stage": "idle",
    "question": None,
    "last_score": 0,
    "avg_score": 0.0,
    "completed": False,
}

current_session_id = None

def init_session(name: str, skills: list):
    """Create a new interview session document in MongoDB."""
    global current_session_id
    if not MONGO_OK or mongo_sessions is None:
        return None
    doc = {
        "name": name,
        "skills": skills,
        "interactions": [],
        "avg_score": 0,
        "created_at": datetime.utcnow(),
    }
    try:
        res = mongo_sessions.insert_one(doc)
        current_session_id = res.inserted_id
        return current_session_id
    except Exception as e:
        print(f"⚠ Failed to create session: {e}")
        return None

def append_interaction(question: str, answer: str, score: int | None):
    """Append an interaction to the current session."""
    if not MONGO_OK or mongo_sessions is None or current_session_id is None:
        return
    try:
        mongo_sessions.update_one(
            {"_id": current_session_id},
            {"$push": {"interactions": {
                "question": question,
                "answer": answer,
                "score": score,
                "ts": datetime.utcnow(),
            }}}
        )
    except Exception as e:
        print(f"⚠ Failed to append interaction: {e}")

def finalize_session(avg_score: float):
    """Update final average score for the session."""
    if not MONGO_OK or mongo_sessions is None or current_session_id is None:
        return
    try:
        mongo_sessions.update_one(
            {"_id": current_session_id},
            {"$set": {"avg_score": avg_score, "completed_at": datetime.utcnow()}}
        )
    except Exception as e:
        print(f"⚠ Failed to finalize session: {e}")

def get_session_history(limit: int = 20):
    """Return latest interview sessions (name, avg_score, count)."""
    if not MONGO_OK or mongo_sessions is None:
        return []
    try:
        docs = list(mongo_sessions.find({}, {
            "name": 1,
            "avg_score": 1,
            "created_at": 1,
            "interactions": {"$slice": 1},
        }).sort("created_at", -1).limit(limit))
        # Convert ObjectId and datetime to strings
        def serialize(d):
            d["_id"] = str(d.get("_id"))
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            return d
        return [serialize(d) for d in docs]
    except Exception as e:
        print(f"⚠ Failed to fetch history: {e}")
        return []

# ========================= INTERVIEW CONTROLLER =========================

class InterviewController:
    """Simple controller to manage interview state for Flask API"""
    
    def __init__(self):
        self.is_running = False
        self.status = "idle"
        self.thread = None
    
    def start_interview(self):
        """Start the interview in a background thread"""
        if self.is_running:
            return False
        
        self.is_running = True
        self.status = "running"
        
        import threading
        self.thread = threading.Thread(target=self._run_interview)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _run_interview(self):
        """Internal method to run the interview"""
        try:
            print("\n✅ Interview thread started successfully")
            main()
            self.status = "completed"
            print("✅ Interview completed successfully")
        except KeyboardInterrupt:
            print("⚠ Interview interrupted by user")
            self.status = "interrupted"
        except Exception as e:
            print(f"❌ Interview error: {e}")
            import traceback
            traceback.print_exc()
            self.status = f"error: {str(e)}"
        finally:
            self.is_running = False
    
    def end_interview(self):
        """End the interview"""
        self.is_running = False
        self.status = "ended"
    
    def get_status(self):
        """Get current interview status for frontend polling."""
        merged = dict(current_status)
        merged["status"] = self.status
        merged["is_running"] = self.is_running
        # ensure defaults
        merged.setdefault("name", None)
        merged.setdefault("stage", "idle")
        merged.setdefault("question", None)
        merged.setdefault("last_score", 0)
        merged.setdefault("avg_score", 0.0)
        merged.setdefault("completed", False)
        return merged


# Global controller instance
interview_controller = InterviewController()

# ========================= HR QUESTION UTIL =========================

def pick_hr_questions(n: int = 10):
    """
    Load HR questions and return up to n random questions across allowed categories.
    Excludes:
      - salary_availability
      - final
    """
    if not HR_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"HR questions file not found at {HR_QUESTIONS_PATH}")

    with open(HR_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {})
    all_q: list[str] = []
    for cat_name, cat in categories.items():
        if cat_name in ("salary_availability", "final"):
            continue
        all_q.extend(cat.get("questions", []))

    if not all_q:
        raise ValueError("No HR questions found in hr_questions.json (after filtering).")

    import random
    random.shuffle(all_q)
    return all_q[:n]


def get_final_question() -> str:
    """
    Get 'Do you have any questions for me?' from hr_questions.json if present,
    otherwise return a default string.
    """
    if not HR_QUESTIONS_PATH.exists():
        return "Before we finish, do you have any questions for me or about this interview?"

    try:
        with open(HR_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        final_cat = data.get("categories", {}).get("final", {})
        questions = final_cat.get("questions", [])
        if questions:
            return questions[0]
    except Exception:
        pass

    return "Before we finish, do you have any questions for me or about this interview?"


# ========================= AUDIO HELPERS =========================

def play_pcm_int16(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    # Lazy-import to avoid startup failure when native audio libs are missing.
    try:
        _np = np if np is not None else __import__("numpy")
        _sd = sd if sd is not None else __import__("sounddevice")
    except Exception:
        print("⚠ Audio playback libraries not available; skipping playback.")
        return

    audio = _np.frombuffer(pcm_bytes, dtype=_np.int16)
    _sd.play(audio, sample_rate)
    _sd.wait()


def record_from_mic(duration_sec: int = ANSWER_RECORD_SECONDS) -> bytes:
    """
    Record from microphone for duration_sec seconds and return WAV bytes.
    Starts immediately after Victus finishes speaking – no extra ENTER/beep.
    Falls back to silent audio if no microphone available.
    """
    print(f"\n🎙 Recording for {duration_sec} seconds... speak whenever you're ready.")
    try:
        # Lazy-import sound libs; if not available, return silent audio
        try:
            _sd = sd if sd is not None else __import__("sounddevice")
        except Exception:
            raise RuntimeError("sounddevice not available")

        _sd.default.samplerate = SAMPLE_RATE
        _sd.default.channels = CHANNELS

        audio = _sd.rec(int(duration_sec * SAMPLE_RATE), dtype="int16")
        _sd.wait()
        print("✅ Recording complete.")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        return buf.getvalue()
    except Exception as e:
        print(f"⚠ Microphone recording failed: {e}. Returning silence.")
        # Return silent audio so interview can continue
        silence = b'\x00' * (SAMPLE_RATE * 2 * CHANNELS * duration_sec)
        return silence


# ========================= LOCAL TTS (NO QUOTA) =========================

def local_tts_say(text: str):
    """Final fallback: OS voice via pyttsx3."""
    if not text or not text.strip():
        return

    print(f"\n🔊 [LOCAL TTS] {text}")
    try:
        if engine_tts is None:
            # pyttsx3 is not available or failed to initialize; skip local TTS.
            print("⚠ engine_tts not available; skipping local TTS.")
            return

        engine_tts.say(text)
        engine_tts.runAndWait()
    except Exception as e:
        print(f"❌ Local TTS failed: {e}")


# ========================= TTS WITH ELEVENLABS PRIMARY =========================

def tts_say(text: str):
    """
    Speak text with this chain:
      1) ElevenLabs TTS (primary)
      2) Gemini TTS
      3) Groq TTS (playai-tts)
      4) Local TTS (pyttsx3)
    """
    global GEMINI_TTS_ENABLED, GROQ_TTS_ENABLED

    if not text or not text.strip():
        return

    print(f"\n🗣 Victus says: {text}")

    # -------- 1) ElevenLabs TTS --------
    try:
        if eleven_tts(text):
            return
        else:
            print("⚠ ElevenLabs TTS did not produce audio. Falling back to Gemini TTS...")
    except Exception as e:
        print(f"❌ ElevenLabs TTS error: {e}")
        print("➡ Falling back to Gemini TTS...")

    # -------- 2) Gemini TTS --------
    if GEMINI_TTS_ENABLED:
        try:
            resp = client_gemini.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=ga_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=ga_types.SpeechConfig(
                        voice_config=ga_types.VoiceConfig(
                            prebuilt_voice_config=ga_types.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    ),
                ),
            )

            audio_bytes = None
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        audio_bytes = inline.data
                        break
                if audio_bytes:
                    break

            if not audio_bytes:
                raise Exception("Gemini TTS returned no audio parts")

            print("🔊 Gemini TTS used.")
            play_pcm_int16(audio_bytes, SAMPLE_RATE)
            return

        except Exception as e:
            print(f"⚠ Gemini TTS failed: {e}")
            GEMINI_TTS_ENABLED = False
            print("➡ Falling back to Groq TTS...")

    # -------- 3) Groq TTS --------
    if GROQ_TTS_ENABLED:
        try:
            speech = client_groq.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=text,
                response_format="wav",
            )

            wav_path = "temp_tts.wav"
            speech.write_to_file(wav_path)

            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                play_pcm_int16(frames, wf.getframerate())
            print("🔊 Groq TTS used.")
            return

        except Exception as e:
            GROQ_TTS_ENABLED = False
            print(f"❌ GROQ TTS failed: {e}")
            print("➡ Falling back to LOCAL TTS...")

    # -------- 4) Local TTS --------
    try:
        local_tts_say(text)
        print("🔊 Local TTS used.")
    except Exception as e:
        print(f"❌ Local TTS also failed: {e}. Interview will continue without audio.")


# ========================= STT WITH ELEVENLABS PRIMARY =========================

def stt_transcribe(wav_bytes: bytes) -> str:
    """
    Speech-to-text chain:
      1) ElevenLabs STT (primary)
      2) Gemini STT
      3) Groq Whisper
    """

    # ---- 1) ElevenLabs STT ----
    try:
        text = eleven_stt(wav_bytes)
        if text:
            print("🔊 ElevenLabs STT used.")
            return text.strip()
    except Exception as e:
        print(f"⚠ ElevenLabs STT failed: {e}")
        print("➡ Falling back to Gemini STT...")

    # ---- 2) Gemini STT ----
    try:
        resp = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                ga_types.Content(
                    parts=[
                        ga_types.Part.from_bytes(
                            data=wav_bytes, mime_type="audio/wav"
                        ),
                        ga_types.Part(
                            text=(
                                "Transcribe this audio into plain text. "
                                "Output only the transcription, no extra explanation."
                            )
                        ),
                    ]
                )
            ],
            config=ga_types.GenerateContentConfig(max_output_tokens=512),
        )

        if resp.text:
            print("🔊 Gemini STT used.")
            return resp.text.strip()

    except Exception as e:
        print(f"⚠ Gemini STT failed: {e}")
        print("➡ Falling back to Groq Whisper...")

    # ---- 3) Groq Whisper STT ----
    file_path = "temp_answer.wav"
    try:
        with open(file_path, "wb") as f:
            f.write(wav_bytes)

        transcription = client_groq.audio.transcriptions.create(
            file=open(file_path, "rb"),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language="en",
        )
        print("🔊 Groq Whisper used.")
        return transcription.text.strip()

    except Exception as e:
        print(f"❌ Whisper STT failed: {e}")
        print("⚠ All STT engines exhausted. Returning empty transcript to continue interview.")

    return ""


# ========================= PASS / SKIP DETECTION =========================

def is_pass_answer(transcript: str) -> bool:
    """
    Check if the candidate wants to pass/skip the question.
    Triggers on phrases like:
      - 'pass this question'
      - 'pass'
      - 'sorry'
    """
    if not transcript:
        return False

    t = transcript.lower()
    if "pass this question" in t:
        return True
    if re.match(r"^\s*pass\b", t):
        return True
    if "sorry" in t:
        return True

    return False


# ========================= NAME & SKILL EXTRACTION =========================

def extract_name_from_text(transcript: str) -> str:
    """
    Simple name extractor from spoken text.
    Examples:
      'My name is Rahul'
      'I am Priya'
      'This is Kartik'
    Fallback: last alpha word.
    """
    if not transcript:
        return ""

    t = transcript.strip()
    m = re.search(r"(?:my name is|i am|this is)\s+([A-Za-z]+)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()

    parts = [p for p in re.split(r"\s+", t) if p.isalpha()]
    if parts:
        return parts[-1].title()

    return t.split()[0].title()


def extract_skills_from_text(transcript: str) -> list:
    """
    Parse spoken skills like 'C, Java, Python, data structures and OS'
    into a normalized skill list.
    """
    if not transcript:
        return []

    t = transcript.lower()
    skills = []

    def add(label):
        if label not in skills:
            skills.append(label)

    if "c++" in t or "cpp" in t:
        add("C++")
    if re.search(r"\bc\b", t):
        add("C")
    if "java" in t:
        add("Java")
    if "python" in t:
        add("Python")
    if "javascript" in t or "js" in t:
        add("JavaScript")
    if "mern" in t:
        add("MERN")
    if "data structure" in t or "dsa" in t:
        add("Data Structures and Algorithms")
    if "operating system" in t or "os" in t:
        add("Operating Systems")
    if "machine learning" in t or "ml" in t:
        add("Machine Learning")
    if "deep learning" in t:
        add("Deep Learning")
    if "database" in t or "sql" in t:
        add("Databases")

    if not skills:
        skills.append("general computer science")

    return skills


def save_candidate_to_mongo(name: str, skills: list, skills_raw: str):
    """Store candidate basic info in MongoDB (HireSense.CandidateData)."""
    if not MONGO_OK or mongo_candidates is None:
        print("⚠ Skipping MongoDB save (not connected).")
        return

    doc = {
        "name": name,
        "skills": skills,
        "skills_raw": skills_raw,
        "created_at": datetime.utcnow(),
    }

    try:
        mongo_candidates.insert_one(doc)
        print(f"✅ Candidate stored in MongoDB: {doc}")
    except Exception as e:
        print(f"⚠ Failed to insert candidate in MongoDB: {e}")


# ========================= MAIN INTERVIEW FLOW =========================

def main():
    print(
        "\n🎤 Victus Voice Interview (HR + Technical + RAG + Mongo)\n"
        "Pure voice conversation. No keyboard input during the flow.\n"
    )

    # ---- Victus intro ----
    tts_say(
        "Hello, I am Victus, your virtual interviewer today. "
        "We will have a short H R round followed by a technical round based on your skills and my textbooks."
    )

    # ===== GET CANDIDATE NAME BY VOICE =====
    tts_say(
        "To begin, please clearly say your name."
    )
    name_audio = record_from_mic(duration_sec=5)
    name_transcript = stt_transcribe(name_audio)
    print("\n📝 Name transcript:", name_transcript)

    candidate_name = extract_name_from_text(name_transcript)
    if not candidate_name:
        candidate_name = "Friend"

    # Update status and init session
    current_status["name"] = candidate_name
    current_status["stage"] = "HR"
    current_status["question"] = None
    current_status["last_score"] = 0
    current_status["avg_score"] = 0.0
    init_session(candidate_name, [])

    tts_say(
        f"Nice to meet you, {candidate_name}. "
        f"I will call you {candidate_name} during this interview."
    )

    # ===== GET SKILLS BY VOICE =====
    tts_say(
        f"{candidate_name}, now tell me which programming languages or technical areas "
        "you are most comfortable with. For example C, Java, Python, data structures, "
        "operating systems or machine learning."
    )
    skills_audio = record_from_mic(duration_sec=8)
    skills_transcript = stt_transcribe(skills_audio)
    print("\n📝 Skills transcript:", skills_transcript)

    skills_list = extract_skills_from_text(skills_transcript)
    skills_str = ", ".join(skills_list)
    tts_say(
        f"Great, I heard that you are comfortable with {skills_str}. "
        "I will keep that in mind for the technical questions."
    )

    # ===== SAVE BASIC PROFILE TO MONGO =====
    save_candidate_to_mongo(candidate_name, skills_list, skills_transcript)
    # attach skills to session if available
    if current_session_id and MONGO_OK and mongo_sessions is not None:
        try:
            mongo_sessions.update_one({"_id": current_session_id}, {"$set": {"skills": skills_list}})
        except Exception:
            pass

    # ================= HR ROUND =================
    print("\n===== HR ROUND (3 non-pass answers, no salary questions) =====")
    tts_say(
        "Let's begin with the H R round. "
        "I will ask you a series of questions. "
        "If you really want to skip one, you can say pass, but I will ask another question instead."
    )

    try:
        hr_pool = pick_hr_questions(n=10)
    except Exception as e:
        print("❌ Failed to load HR questions:", e)
        tts_say(
            "I could not load H R questions properly, "
            "so we will jump directly to the technical round."
        )
        hr_pool = []

    hr_answers = []
    hr_answered = 0
    hr_index = 0
    hr_max_loops = 30  # safety

    while hr_answered < HR_REQUIRED and hr_max_loops > 0:
        hr_max_loops -= 1

        if hr_index >= len(hr_pool):
            # refill pool if exhausted
            try:
                hr_pool = pick_hr_questions(n=10)
                hr_index = 0
            except Exception as e:
                print("❌ Could not refill HR questions:", e)
                break

        if not hr_pool:
            break

        q = hr_pool[hr_index]
        hr_index += 1

        print(f"\n[HR Q] {q}")
        tts_say(q)
        current_status["stage"] = "HR"
        current_status["question"] = q

        # Directly start listening
        wav_bytes = record_from_mic(duration_sec=ANSWER_RECORD_SECONDS)

        print("\n⏳ Transcribing your H R answer...")
        transcript = stt_transcribe(wav_bytes)
        print("\n📝 Your HR Answer:\n", transcript)

        if is_pass_answer(transcript):
            tts_say(
                "Alright, we will not consider this question. "
                "I will ask you a different one."
            )
            hr_answers.append({"question": q, "answer": transcript, "skipped": True})
            append_interaction(q, transcript, None)
            continue

        hr_answers.append({"question": q, "answer": transcript, "skipped": False})
        append_interaction(q, transcript, None)
        hr_answered += 1

        if transcript.strip():
            tts_say("Thank you for sharing that.")
        else:
            tts_say(
                "I could not clearly hear your answer, but let's keep going."
            )

    # ================= TECHNICAL ROUND =================
    print("\n===== TECHNICAL ROUND (3 evaluated answers from RAG) =====")
    tts_say(
        "Now we will move to the technical round. "
        "These questions are based on your skills and the textbooks I have read. "
        "If you say pass or sorry, I will ask another question, but you still need to answer three questions."
    )

    topic_hint = skills_str if skills_list else "computer science"

    tech_results = []
    total_score = 0
    tech_answered = 0
    questions_buffer = []
    q_index = 0
    tech_max_loops = 50  # safety

    while tech_answered < TECH_REQUIRED and tech_max_loops > 0:
        tech_max_loops -= 1

        # refill buffer if empty or exhausted
        if q_index >= len(questions_buffer):
            try:
                print("\n⏳ Fetching technical questions from RAG...")
                # Lazy import to avoid heavy ML imports at server startup
                try:
                    from services.interview_service import start_interview
                except Exception as _e:
                    print(f"⚠ start_interview import failed: {_e}")
                    start_interview = None

                if start_interview is None:
                    raise RuntimeError("start_interview not available (missing dependencies)")

                interview_data = start_interview(
                    topic=topic_hint,
                    num_questions=3,
                    random_from_rag=True,
                )
            except Exception as e:
                print(f"❌ Error starting technical interview: {e}")
                tts_say(
                    "I encountered an error while generating more technical questions. "
                    "We will stop here."
                )
                break

            if not isinstance(interview_data, dict):
                print("❌ start_interview() returned invalid data:", interview_data)
                tts_say(
                    "I received invalid data from the question generator. "
                    "We will stop here."
                )
                break

            questions_buffer = interview_data.get("questions") or []
            global_sources = interview_data.get("global_sources", [])
            q_index = 0

            if not questions_buffer:
                print("❌ No technical questions were generated by the backend.")
                print("   Raw interview_data:", interview_data)
                tts_say(
                    "No technical questions were generated. "
                    "We will stop here."
                )
                break

        q = questions_buffer[q_index]
        q_index += 1

        qid = q.get("id")
        qtext = q.get("question")
        ideal = q.get("ideal_answer")
        q_sources = q.get("sources", global_sources)

        print(f"\n[TECH Q] {qtext}")
        print("\n📚 Sources:")
        for src in q_sources:
            print(" -", src)

        tts_say(qtext)
        current_status["stage"] = "TECH"
        current_status["question"] = qtext

        # Listen immediately
        wav_bytes = record_from_mic(duration_sec=ANSWER_RECORD_SECONDS)

        print("\n⏳ Transcribing your answer...")
        transcript = stt_transcribe(wav_bytes)
        print("\n📝 You said:\n", transcript)

        if is_pass_answer(transcript):
            tts_say(
                "Okay, I will not evaluate this one. "
                "Let me ask you another technical question."
            )
            continue  # do not increment tech_answered

        if not transcript.strip():
            tts_say(
                "I could not clearly hear your answer, but I will still try to evaluate what I got."
            )

        print("\n⏳ Evaluating your answer...")
        try:
            # Lazy import evaluation logic
            try:
                from services.interview_service import evaluate_answer
            except Exception as _e:
                print(f"⚠ evaluate_with_gemini import failed: {_e}")
                evaluate_answer = None

            if evaluate_answer is None:
                raise RuntimeError("evaluate_answer not available (missing deps)")

            eval_res = evaluate_answer(
                question_id=qid,
                question_text=qtext,
                ideal_answer=ideal,
                user_answer=transcript,
            )
        except Exception as e:
            print(f"❌ Error in evaluation for a technical question: {e}")
            tts_say(
                "There was an error while evaluating this answer. "
                "We will move to another question."
            )
            continue

        score = eval_res.get("score", 0)
        feedback = eval_res.get("feedback", "")
        missing_points = eval_res.get("missing_points", [])
        eval_sources = eval_res.get("sources", [])
        reliability = eval_res.get("reliability_score", 0)
        grounded = eval_res.get("grounded_in_context", False)

        total_score += score
        tech_answered += 1
        tech_results.append(eval_res)
        append_interaction(qtext, transcript, score)

        # Update live status
        try:
            avg_now = total_score / tech_answered if tech_answered > 0 else 0
        except Exception:
            avg_now = 0
        current_status["last_score"] = int(score)
        current_status["avg_score"] = float(avg_now)

        print("\n🎯 RESULT FOR THIS QUESTION")
        print("---------------------------")
        print(f"Score: {score}/100")
        print(f"Reliability: {reliability}/100")
        print(f"Grounded in context: {grounded}")
        print(f"Feedback: {feedback}")
        if missing_points:
            print("Missing points:")
            for mp in missing_points:
                print("  -", mp)

        if eval_sources:
            print("\n📚 Evaluation context was taken from:")
            for src in eval_sources:
                print(" -", src)

        tts_say(f"For this question, your score is {score} out of 100. {feedback}")

    # ================= SUMMARY & FINAL QUESTION =================
    if tech_answered > 0:
        avg_score = total_score / tech_answered
    else:
        avg_score = 0

    summary = (
        f"{candidate_name}, this brings us to the end of your technical round. "
        f"Based on the answers I evaluated, your average score is {int(avg_score)} out of 100."
    )

    print("\n===== INTERVIEW SUMMARY =====")
    print(summary)
    tts_say(summary)
    finalize_session(avg_score)

    # Final “Any questions for us?” question
    final_q = get_final_question()
    print("\n[FINAL] ", final_q)
    tts_say(
        "Before we finish, I have one last question for you."
    )
    tts_say(final_q)
    tts_say("You can speak freely now, and I will listen.")

    final_audio = record_from_mic(duration_sec=15)
    final_transcript = stt_transcribe(final_audio)
    print("\n📝 Candidate's final question/comment:\n", final_transcript)

    tts_say(
        "Thank you for your time and for taking this mock interview with me. "
        "This session is now complete. I wish you all the best for your real interviews."
    )

    # Mark as completed for frontend
    current_status["completed"] = True


if __name__ == "__main__":
    main()
