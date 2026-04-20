# policajt

Motion-activated camera watcher. Checks the PC camera once per second and,
on detected motion, plays an MP3 and sends an email every second for 5
seconds.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your SMTP credentials
```

Place an `alert.mp3` in the project directory (or set `MP3_PATH`).

For Gmail, create an App Password at https://myaccount.google.com/apppasswords
and use it as `SMTP_PASSWORD`.

## Run

```bash
python policajt.py
```

On macOS you'll be prompted to grant camera and microphone access to your
terminal the first time you run it.

## Tuning

- `MOTION_THRESHOLD` — minimum % of changed pixels to trigger (default 5).
- `PIXEL_DIFF_THRESHOLD` — per-pixel brightness diff to count as changed.
- `CHECK_INTERVAL_SEC` — how often to sample the camera (default 1s).
- `ALERT_DURATION_SEC` — how long to keep alerting after a trigger (default 5s).