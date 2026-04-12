# Deploy (Render)

This folder is a self-contained copy of the Telegram bot for **Render** (or any Linux host with Python 3.12+).

## What’s inside

- `bot.py`, `default_driving.py`, `whitelist.py`
- `requirements.txt`, `runtime.txt`
- `data/` — `allowed_chats.json`, `default_driving.mp4`, `default_driving_meta.json` (bundled defaults)
- `.env.example` — copy secrets into Render **Environment** (do not commit real `.env`)

## Render: Background Worker

1. Push this repo to GitHub/GitLab (or only the `deploy/` folder as its own repo).
2. Render → **New** → **Background Worker**.
3. Connect the repo. If the repo root is the parent project, set **Root Directory** to `deploy`.
4. **Build command:** `pip install --upgrade pip && pip install -r requirements.txt`
5. **Start command:** `python bot.py`
6. **Environment:** add at least `TELEGRAM_BOT_TOKEN` (and optional vars from `.env.example`). Use **Secret** for tokens/passwords.

`render.yaml` is optional: **Blueprint** → paste file, or attach from repo root (move/copy `render.yaml` to repo root if Render expects it there).

## Notes

- **Ephemeral disk:** `data/` on Render resets on redeploy unless you add a **persistent disk** and mount it where `data/` lives (advanced). For allow-list + default video persistence, use env `ALLOWED_CHAT_IDS` and re-upload default video after deploy, or attach a disk.
- **Free tier:** workers may not be ideal for 24/7 bots; pick a plan that stays awake.
- **OpenCV:** `opencv-python-headless` is in `requirements.txt` (no GUI on servers).

## Local test (same as project root)

```bash
cd deploy
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env   # fill TELEGRAM_BOT_TOKEN
python bot.py
```
