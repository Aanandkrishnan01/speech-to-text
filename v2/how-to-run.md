# How to Run (Docker)

Step-by-step Docker walkthrough. Follow top to bottom — every command is
copy-pasteable and they run in order.

---

## Step 1 — Install Docker Desktop (one-time)

Skip this step if `docker --version` already prints something.

| OS      | Install                                               |
| ------- | ----------------------------------------------------- |
| macOS   | https://www.docker.com/products/docker-desktop/       |
| Windows | https://www.docker.com/products/docker-desktop/       |
| Linux   | https://docs.docker.com/engine/install/               |

After installing, **launch Docker Desktop** and wait until its tray icon
shows "Docker Desktop is running". Confirm:

```bash
docker info
```

If you see daemon info (not an error), continue.

---

## Step 2 — Get the project

```bash
git clone <your-repo-url>
cd Speech-To-Text-STT-
```

You should see `Dockerfile`, `docker-compose.yml`, `config.json`, etc.

---

## Step 3 — Create your `.env` file

```bash
cp .env.example .env
```

Leave `HF_TOKEN` blank — it's optional and only needed for gated HuggingFace
models. The default backends (NeMo + Qwen3-ASR) don't need it.

---

## Step 4 — Build the Docker image (first time, ~10–20 min)

```bash
docker compose build
```

This downloads the Python base image, installs system libs (ffmpeg,
portaudio, sox, libsndfile), and `pip install`s every dependency
(torch, NeMo, Qwen3-ASR, FastAPI, …).

You only do this once. Subsequent code changes rebuild in ~30 seconds
because the `pip install` layer is cached.

When it's done you'll see something like:

```
stt:latest  Built
```

Verify the image is there:

```bash
docker images stt
```

---

## Step 5 — Verify everything is wired up correctly

```bash
docker compose run --rm stt python main.py verify
```

Expected output (last lines):

```
✓ Configuration loaded
✓ All dependencies installed
✓ FFmpeg installed
⚠ No audio devices accessible: ...
  (expected in Docker on Mac/Win — batch + web modes still work)
✓ Setup verification complete!
```

The "No audio devices" warning is **normal** in Docker — that's why we use
the **web server mode** for realtime (the browser handles the mic, not the
container).

---

## Step 6a — Run the web UI (most common path)

```bash
docker compose up stt-web
```

What happens:

1. Container starts.
2. Models load. **First run downloads ~3.5 GB** of weights into the
   `model-cache` Docker volume — this takes a few minutes.
3. You'll see:
   ```
   Application startup complete.
   Uvicorn running on http://0.0.0.0:8000
   ```

Open your browser:

> **http://localhost:8000**

Click **Start mic** → allow the permission prompt → speak → watch transcripts
appear with speaker labels.

> ⚠️ **Use `localhost`, not your LAN IP.** Browsers block mic access on
> non-secure origins. `http://localhost:8000` works; `http://192.168.x.x:8000`
> does not.

**Stop the server**: press `Ctrl+C` in the terminal, then:

```bash
docker compose down
```

---

## Step 6b — Transcribe a saved audio file (batch mode)

Drop your file in `./audio/` (it shows up as `/audio/` inside the container):

```bash
mkdir -p audio
cp ~/path/to/meeting.mp3 audio/meeting.mp3

docker compose run --rm stt python main.py batch /audio/meeting.mp3
```

Output:
- Transcript saved to `./transcripts/meeting_<backend>_<model>_<timestamp>.txt`
- Logs in `./logs/`

You can run multiple files:
```bash
docker compose run --rm stt python main.py batch /audio/file1.mp3
docker compose run --rm stt python main.py batch /audio/file2.wav
```

---

## Step 7 — Pick your ASR backend (optional)

Edit `config.json` on your host (the change is picked up next run — no rebuild
needed):

```jsonc
{
  "asr_backend": "qwen",   // change to "nemo" for English-only / faster / smaller
  ...
}
```

| Backend | Default models                                  | Best for                                          |
| ------- | ----------------------------------------------- | ------------------------------------------------- |
| `qwen`  | Qwen/Qwen3-ASR-0.6B (rt) / 1.7B (batch)         | Multilingual, best accuracy, ~3.5 GB download     |
| `nemo`  | stt_en_conformer_ctc_small / large              | English-only, smaller (~200–500 MB), faster on CPU |

After editing, restart the container:
```bash
docker compose down
docker compose up stt-web
```

---

## Step 8 — Where things land on your host

| What             | Path                                  |
| ---------------- | ------------------------------------- |
| Transcripts      | `./transcripts/`                      |
| Application logs | `./logs/`                             |
| Audio inputs     | `./audio/`                            |
| Model weights    | Docker volume `model-cache` (persistent across container runs) |
| Config           | `./config.json` (edited on host)      |

You can `tail -f logs/stt_$(date +%Y%m%d).log` while the container is running
to watch the full debug stream.

---

## Step 9 — Updating after a `git pull`

```bash
git pull
docker compose build              # 30 sec if only code changed; 5–10 min if requirements.txt changed
docker compose up stt-web
```

---

## Step 10 — Cleaning up

| Goal                                                   | Command                              |
| ------------------------------------------------------ | ------------------------------------ |
| Stop the running web server                            | Ctrl+C, then `docker compose down`   |
| Stop AND wipe downloaded models (force re-download)    | `docker compose down -v`             |
| Remove the built image entirely                        | `docker rmi stt:latest`              |
| Reclaim disk from old layers                           | `docker system prune -af`            |

---

## Common errors

### `Cannot connect to the Docker daemon`
Docker Desktop isn't running. Launch it, wait for the tray icon, retry.

### `ERROR: Cannot install ... accelerate==1.13.0`
Stale `requirements.txt`. Run `git pull` and rebuild.

### `mic denied: Cannot read properties of undefined (reading 'getUserMedia')`
You opened a LAN IP instead of `localhost`. Use **http://localhost:8000**
exactly. The mic API needs a "secure context" — `localhost` qualifies, LAN
IPs don't.

### Web UI loads but no transcripts appear when I speak
Check the **VU meter** under the controls — if the green bar moves when you
speak, audio is reaching the server. If it doesn't move:
- Browser denied the mic permission. Click the mic icon in the URL bar to
  re-allow.
- Wrong input device selected by the OS. Check Mac System Settings →
  Sound → Input.

### "All NeMo models loaded" — but I picked `qwen`
NeMo VAD + titanet are required for **diarization** regardless of which ASR
backend you pick. Only the transcription step uses your selected backend.
The startup line will say `(ASR: Qwen3-ASR, diarization: NeMo VAD + titanet)`
to make this clear.

### First run is very slow
First run downloads ~3.5 GB of model weights from HuggingFace. Subsequent
runs reuse them from the `model-cache` volume and start in seconds.

---

## Quick reference

```bash
# One-time
docker compose build

# Run web UI
docker compose up stt-web                      # then open http://localhost:8000

# Run batch
docker compose run --rm stt python main.py batch /audio/file.mp3

# Verify
docker compose run --rm stt python main.py verify

# Stop
Ctrl+C
docker compose down
```
