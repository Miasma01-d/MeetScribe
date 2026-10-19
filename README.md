# MeetScribe: local-first AI meeting note-taker

A beginner-friendly prototype for live audio transcription, local AI summaries, a Streamlit dashboard, and optional Google Docs export.

## Important limitations

- This app captures an audio **input device** selected on your Linux machine. The microphone captures your voice. To capture remote meeting participants, choose a system-audio monitor/loopback input if your PulseAudio/PipeWire setup exposes one. Device availability varies by Linux setup.
- This is not a Google Meet bot and does not bypass Google Meet restrictions. Use only audio you are authorized to capture, tell participants, and obtain required consent.
- Live transcription is chunked and may lag. This prototype does not identify speakers.
- Google Meet's REST API can retrieve eligible transcripts after meetings; it is not a general live audio feed. Real-time Meet Media API access has restrictions.
- This prototype stores transcript text locally under `meetings/`. Protect that folder and do not commit tokens or credentials to source control.

## 1. Install Linux audio dependencies

On Linux Mint / Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip portaudio19-dev libasound2-dev
```

## 2. Create a virtual environment

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

If `faster-whisper` fails to install, check that your Python version is supported and that the virtual environment is active.

## 3. Install Ollama and download a small model

Install Ollama from its official Linux instructions: https://ollama.com/download/linux

Then start it (the installer may already start the service) and run:

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

After confirming it works, exit the interactive model with `/bye`. Ollama normally serves its local API at `http://localhost:11434`.

If you want to use a different local model, set `OLLAMA_MODEL` before launching the app, or change the model name in the sidebar.

## 4. Launch MeetScribe

```bash
source .venv/bin/activate
streamlit run app.py
```

A browser tab should open, usually at http://localhost:8501.

The first time you start recording, Faster-Whisper downloads the `base` speech model. This requires an internet connection once. Transcription then runs locally.

## 5. Select the correct audio input

- Select your microphone to transcribe your own speech.
- To capture both sides of a Meet call, use your Linux sound settings to find a monitor/loopback source (often named something like `Monitor of ...`). You may need to install or configure a PipeWire/PulseAudio loopback/monitor source.
- If no monitor source appears, use an authorized external recording/transcription source or the transcript provided by Google Meet where available. Do not try to circumvent platform controls.
- Use headphones and test with a short, non-sensitive call to avoid feedback.

The transcript updates in chunks of about eight seconds, plus processing time. Click **Stop** when finished, then **Generate / refresh notes**. Edit the notes if needed and download them or create a Google Doc.

## 6. Set up Google Docs export (optional)

1. Open https://console.cloud.google.com/ and create/select a Google Cloud project.
2. Enable **Google Docs API** for the project.
3. Configure the OAuth consent screen. For a personal test app, use the appropriate external/internal option and add your Google account as a test user if required.
4. Create an OAuth Client ID of type **Desktop app** and download the JSON file.
5. Save the downloaded file in this project folder as `credentials.json` (next to `app.py`). Keep it private.
6. Click **Create Google Doc** in MeetScribe. A browser window will ask you to sign in and authorize access. The app saves `token.json` locally for later use.

The app creates a Google Doc in the signed-in account. It uses the Google Docs API scope. If you revoke access, remove `token.json` and authorize again. Never share `credentials.json` or `token.json`.

## 7. Common issues

### `PortAudio` / `sounddevice` error

Install `portaudio19-dev`, restart the terminal, reactivate `.venv`, and rerun the app. Check detected devices with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Ollama connection error

Check the service and model:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### Transcription is slow

Try a smaller Whisper model by changing `WhisperModel("base", ...)` to `WhisperModel("tiny", ...)` in `app.py`. Accuracy may decrease. Closing other applications can help.

### Other meeting participants are not transcribed

The app may be listening to the microphone rather than system audio. Select a monitor/loopback input in the sidebar. Audio-device names and availability depend on your sound server configuration.

## Architecture and next steps

Current flow:

1. `sounddevice` captures mono audio at 16 kHz.
2. Audio is processed in approximately eight-second chunks.
3. Faster-Whisper transcribes each chunk locally.
4. The transcript is sent to Ollama only when you ask to generate notes; Ollama runs locally.
5. Notes can be edited, downloaded, or exported to Google Docs.

Recommended upgrades after this prototype works:

- Add a pause/resume control and meeting-specific folders.
- Save transcripts continuously, not only when requested.
- Improve chunk buffering and audio resampling for devices that do not support 16 kHz directly.
- Add speaker diarization and transcript timestamps.
- Add a supported Google Meet REST API connector to import completed transcripts after meetings, with OAuth scopes and permissions configured correctly.
- Add tests, better error handling, and encrypted storage before using sensitive business meetings.
