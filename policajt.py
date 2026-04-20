"""Motion-activated camera watcher.

Watches the PC camera once per second. When motion is detected, it plays
an MP3 alert and sends an email every second for 5 seconds.

Configuration is read from environment variables (or a local .env file if
python-dotenv is installed). See README.md for details.
"""

from __future__ import annotations

import logging
import os
import shutil
import smtplib
import ssl
import subprocess
import sys
import threading
import time
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


LOG = logging.getLogger("policajt")

# --- Configuration -----------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CHECK_INTERVAL_SEC = float(os.getenv("CHECK_INTERVAL_SEC", "1.0"))
ALERT_DURATION_SEC = float(os.getenv("ALERT_DURATION_SEC", "5.0"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "5.0"))  # % of changed pixels
PIXEL_DIFF_THRESHOLD = int(os.getenv("PIXEL_DIFF_THRESHOLD", "25"))
MP3_PATH = os.getenv("MP3_PATH", "alert.mp3")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "")


# --- Audio -------------------------------------------------------------------
_AUDIO_PLAYER: list[str] | None = None
_AUDIO_PROC: subprocess.Popen | None = None


def init_audio(mp3_path: str) -> bool:
    global _AUDIO_PLAYER
    if not Path(mp3_path).is_file():
        LOG.warning("MP3 file not found at %s; audio alerts disabled", mp3_path)
        return False
    for cmd in (
        ["afplay"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ["mpg123", "-q"],
    ):
        if shutil.which(cmd[0]):
            _AUDIO_PLAYER = cmd
            LOG.info("Using audio player: %s", cmd[0])
            return True
    LOG.warning("No audio player found (afplay/ffplay/mpg123); audio disabled")
    return False


def play_mp3(mp3_path: str) -> None:
    global _AUDIO_PROC
    if _AUDIO_PLAYER is None:
        return
    try:
        if _AUDIO_PROC is not None and _AUDIO_PROC.poll() is None:
            _AUDIO_PROC.terminate()
        _AUDIO_PROC = subprocess.Popen(
            _AUDIO_PLAYER + [mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        LOG.error("Audio playback failed: %s", exc)


# --- Email -------------------------------------------------------------------
def send_email(subject: str, body: str) -> None:
    if not (SMTP_USER and SMTP_PASSWORD and EMAIL_TO):
        LOG.warning("SMTP credentials or recipient missing; skipping email")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        LOG.info("Email sent to %s", EMAIL_TO)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to send email: %s", exc)


def send_email_async(subject: str, body: str) -> None:
    threading.Thread(
        target=send_email, args=(subject, body), daemon=True
    ).start()


# --- Motion detection --------------------------------------------------------
def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def detect_motion(prev: np.ndarray, curr: np.ndarray) -> bool:
    diff = cv2.absdiff(prev, curr)
    _, thresh = cv2.threshold(
        diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    changed_ratio = (np.count_nonzero(thresh) / thresh.size) * 100.0
    LOG.debug("Changed pixels: %.2f%%", changed_ratio)
    return changed_ratio >= MOTION_THRESHOLD


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    # Flush buffer to get the most recent frame.
    for _ in range(3):
        cap.grab()
    ok, frame = cap.read()
    return frame if ok else None


# --- Main loop ---------------------------------------------------------------
def run() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    audio_ok = init_audio(MP3_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        LOG.error("Cannot open camera index %d", CAMERA_INDEX)
        return 1

    LOG.info("Watching camera %d (Ctrl+C to stop)...", CAMERA_INDEX)

    prev_frame = None
    alert_until = 0.0
    last_alert_tick = 0.0

    try:
        while True:
            loop_start = time.monotonic()

            frame = grab_frame(cap)
            if frame is None:
                LOG.warning("Failed to read frame")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            processed = preprocess(frame)

            now = time.monotonic()
            if prev_frame is not None and detect_motion(prev_frame, processed):
                if now >= alert_until:
                    LOG.info("Motion detected — starting %.0fs alert",
                             ALERT_DURATION_SEC)
                    alert_until = now + ALERT_DURATION_SEC
                    last_alert_tick = 0.0  # force immediate tick

            # Fire alert once per second while active.
            if now < alert_until and now - last_alert_tick >= 1.0:
                last_alert_tick = now
                if audio_ok:
                    play_mp3(MP3_PATH)
                send_email_async(
                    subject="Motion detected",
                    body=f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                )

            prev_frame = processed

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, CHECK_INTERVAL_SEC - elapsed))
    except KeyboardInterrupt:
        LOG.info("Stopping.")
    finally:
        cap.release()
        if _AUDIO_PROC is not None and _AUDIO_PROC.poll() is None:
            _AUDIO_PROC.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(run())
