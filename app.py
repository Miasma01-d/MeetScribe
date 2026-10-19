import os
import io
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import streamlit as st
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "meetings"
DATA_DIR.mkdir(exist_ok=True)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
CHUNK_SECONDS = 8

st.set_page_config(page_title="MeetScribe", page_icon="📝", layout="wide")
st.title("📝 MeetScribe")
st.caption("Local-first live meeting transcription and AI notes. Use only with appropriate participant consent.")

# Shared background state for this Streamlit process.
if "engine" not in st.session_state:
    st.session_state.engine = {
        "running": False,
        "stop_event": threading.Event(),
        "audio_queue": queue.Queue(),
        "transcript": [],
        "status": "Ready",
        "error": "",
        "meeting_name": "Meeting notes",
        "started_at": None,
        "audio_device": None,
        "worker": None,
    }

engine = st.session_state.engine


def list_input_devices():
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0:
            devices.append({"id": i, "name": dev["name"], "channels": int(dev["max_input_channels"])})
    return devices


def save_transcript():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in engine["meeting_name"]).strip().replace(" ", "_") or "meeting"
    path = DATA_DIR / f"{safe_name}_{stamp}.txt"
    lines = [f"{engine['meeting_name']}", f"Captured: {engine['started_at'] or datetime.now().isoformat(timespec='seconds')}", "", "TRANSCRIPT", ""]
    lines += [f"[{item['time']}] {item['text']}" for item in engine["transcript"]]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def prepare_whisper_audio(audio):
    # Convert 48 kHz microphone audio to Whisper's expected 16 kHz.
    return resample_poly(
        np.asarray(audio, dtype=np.float32), 1, 3
    ).astype(np.float32)


def transcribe_worker(device_id, stop_event, audio_queue, state):
    try:
        state["status"] = "Loading speech recognition model (first run downloads it)..."
        # CPU-friendly starter model. Use compute_type=int8 to reduce memory use.
        whisper = WhisperModel("base", device="cpu", compute_type="int8")
        state["status"] = "Listening. Speak clearly and keep the meeting volume audible."
        audio_buffer = []
        buffer_samples = 0
        stream = sd.InputStream(
            device=device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=0,
            callback=lambda indata, frames, time_info, status: audio_queue.put(indata[:, 0].copy()),
        )
        with stream:
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                audio_buffer.append(chunk)
                buffer_samples += len(chunk)
                if buffer_samples >= SAMPLE_RATE * CHUNK_SECONDS:
                    audio = np.concatenate(audio_buffer).astype(np.float32)
                    audio_buffer = []
                    buffer_samples = 0
                    audio = prepare_whisper_audio(audio)
                    segments, _info = whisper.transcribe(audio, language=None, vad_filter=True, beam_size=1)
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    if text:
                        state["transcript"].append({"time": datetime.now().strftime("%H:%M:%S"), "text": text})
        # Transcribe any final partial chunk after stopping.
        if audio_buffer:
            audio = np.concatenate(audio_buffer).astype(np.float32)
            audio = prepare_whisper_audio(audio)
            segments, _info = whisper.transcribe(audio, language=None, vad_filter=True, beam_size=1)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                state["transcript"].append({"time": datetime.now().strftime("%H:%M:%S"), "text": text})
        state["status"] = "Stopped"
    except Exception as exc:
        state["error"] = str(exc)
        state["status"] = "Error"
    finally:
        state["running"] = False


def get_transcript_text():
    return "\n".join(f"[{x['time']}] {x['text']}" for x in engine["transcript"])


def generate_notes(transcript, model):
    if not transcript.strip():
        return "No transcript yet. Start recording and let the assistant capture some speech first."
    prompt = f"""You are a careful meeting secretary. Use only the transcript below. Do not invent names, decisions, owners, or deadlines. Mark details as 'Not specified' when missing. If speech is unclear, say so.

Create concise, readable meeting minutes with these headings:
1. Meeting summary
2. Key discussion points
3. Decisions made
4. Action items (task | owner | deadline)
5. Open questions / follow-ups

TRANSCRIPT:
{transcript[-24000:]}
"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "No notes returned.")


def create_google_doc(title, body_text):
    """Create a Google Doc using OAuth Desktop credentials in credentials.json."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/documents"]
    token_path = APP_DIR / "token.json"
    credentials_path = APP_DIR / "credentials.json"
    if not credentials_path.exists():
        raise FileNotFoundError("credentials.json not found. Follow the Google Docs setup steps in README.md.")
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().create(body={"title": title}).execute()
    doc_id = doc["documentId"]
    service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": body_text}}]},
    ).execute()
    return f"https://docs.google.com/document/d/{doc_id}/edit"


with st.sidebar:
    st.header("Meeting setup")
    meeting_name = st.text_input("Meeting name", value=engine["meeting_name"])
    engine["meeting_name"] = meeting_name
    st.caption("Select your microphone to capture your voice. To capture other participants, select a system-audio monitor/loopback device if your Linux audio setup exposes one.")
    try:
        input_devices = list_input_devices()
    except Exception as e:
        input_devices = []
        st.error(f"Could not list audio devices: {e}")
    if input_devices:
        labels = [f"{d['id']}: {d['name']} ({d['channels']} channel(s))" for d in input_devices]
        selected_label = st.selectbox("Audio input", labels)
        selected_device = input_devices[labels.index(selected_label)]["id"]
    else:
        selected_device = None
        st.warning("No audio input devices detected. Check Linux sound permissions and devices.")
    st.divider()
    st.caption("Speech recognition runs locally. AI summaries use Ollama locally when installed.")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Live transcript")
    if engine["running"]:
        st.info(engine["status"])
    elif engine["error"]:
        st.error(engine["error"])
    else:
        st.caption(engine["status"])
    transcript_placeholder = st.empty()
    transcript = get_transcript_text()
    transcript_placeholder.code(transcript[-12000:] if transcript else "Transcript will appear here after recording starts.", language=None)
    col_start, col_stop, col_save = st.columns(3)
    with col_start:
        if st.button("▶ Start", disabled=engine["running"] or selected_device is None, use_container_width=True):
            engine["error"] = ""
            engine["transcript"] = []
            engine["audio_queue"] = queue.Queue()
            engine["stop_event"] = threading.Event()
            engine["started_at"] = datetime.now().isoformat(timespec="seconds")
            engine["running"] = True
            engine["status"] = "Starting..."
            worker = threading.Thread(target=transcribe_worker, args=(selected_device, engine["stop_event"], engine["audio_queue"], engine), daemon=True)
            engine["worker"] = worker
            worker.start()
            st.rerun()
    with col_stop:
        if st.button("■ Stop", disabled=not engine["running"], use_container_width=True):
            engine["stop_event"].set()
            engine["status"] = "Stopping..."
            st.rerun()
    with col_save:
        if st.button("Save transcript", disabled=not engine["transcript"], use_container_width=True):
            saved_path = save_transcript()
            st.success(f"Saved to {saved_path}")

with right:
    st.subheader("AI meeting notes")
    if "notes" not in st.session_state:
        st.session_state.notes = ""
    if st.button("✨ Generate / refresh notes", use_container_width=True):
        try:
            with st.spinner("Asking local AI to organize the transcript..."):
                st.session_state.notes = generate_notes(get_transcript_text(), model_name)
                st.session_state["notes_editor"] = st.session_state.notes
        except requests.exceptions.ConnectionError:
            st.error("Ollama isn't responding. Install/start Ollama and pull the model listed in README.md.")
        except Exception as e:
            st.error(f"Could not generate notes: {e}")
    if "notes_editor" not in st.session_state:
        st.session_state["notes_editor"] = st.session_state.notes
    notes = st.text_area("Editable notes", height=360, key="notes_editor")
    st.session_state.notes = notes
    st.download_button("Download notes (.txt)", data=notes, file_name="meeting_notes.txt", mime="text/plain", disabled=not notes.strip(), use_container_width=True)
    if st.button("Create Google Doc", disabled=not notes.strip(), use_container_width=True):
        try:
            with st.spinner("Connecting to Google Docs..."):
                url = create_google_doc(engine["meeting_name"], notes)
            st.success("Google Doc created.")
            st.markdown(f"[Open your Google Doc]({url})")
        except Exception as e:
            st.error(f"Google Docs export failed: {e}")

st.divider()
st.caption("Prototype limitations: transcription is chunked and may lag; speaker identification is not implemented; audio capture depends on the selected Linux input device. Confirm consent and test on a non-sensitive meeting before use.")

# Refresh the UI while a capture session is active.
if engine["running"]:
    time.sleep(1.0)
    st.rerun()
