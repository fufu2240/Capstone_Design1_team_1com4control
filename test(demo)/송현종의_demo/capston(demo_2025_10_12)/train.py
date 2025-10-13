import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import time
import librosa
from tensorflow.keras.models import load_model

# -------------------------------
# 모델 불러오기 (1초 모델)
# -------------------------------
model = load_model(r"C:\capston\sound_event_model_1s.h5")

# 클래스 라벨
labels = [
    "cry", "cough", "fall", "knock", "noise",
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

# -------------------------------
# 오디오 설정
# -------------------------------
samplerate = 22050
duration = 2   # 2초 읽고 앞쪽 1초만 사용
blocksize = samplerate * duration
running = False
threshold = 0.01   # 🎯 기존 0.02 → 0.01로 낮춤

# -------------------------------
# JBL / Hands-Free 장치 제외한 기본 입력 장치 찾기
# -------------------------------
def get_default_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if dev['max_input_channels'] > 0:
            if "jbl" in name or "hands-free" in name:
                continue  # ❌ JBL/Hands-Free 장치는 제외
            print(f"🎯 사용 가능 입력 장치 선택: {i} - {dev['name']}")
            return i
    raise RuntimeError("❌ 사용할 수 있는 입력 장치가 없습니다 (JBL/Hands-Free 제외됨).")

DEVICE_INDEX = get_default_device()

# -------------------------------
# 전처리 (학습 환경과 동일)
# -------------------------------
def preprocess_audio(audio, sr=22050):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < 43:
        pad = 43 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode="constant")
    else:
        mel_db = mel_db[:, :43]

    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    mel_db = mel_db[..., np.newaxis]   # (128,43,1)
    mel_db = mel_db[np.newaxis, ...]   # (1,128,43,1)
    return mel_db

def predict_sound(audio):
    x = preprocess_audio(audio)
    pred = model.predict(x, verbose=0)[0]
    return pred

# -------------------------------
# Tkinter GUI
# -------------------------------
root = tk.Tk()
root.title("실시간 사운드 분류 (threshold=0.01)")

title = tk.Label(root, text="실시간 사운드 인식", font=("맑은 고딕", 18, "bold"))
title.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

status_label = tk.Label(root, text="수집 대기", font=("맑은 고딕", 12))
status_label.pack(pady=5)

rms_label = tk.Label(root, text=f"RMS: 0.00000 | threshold: {threshold:.5f}", font=("맑은 고딕", 10))
rms_label.pack()

result_labels = {}
for lab in labels:
    l = tk.Label(root, text=f"{lab}: 0%", font=("맑은 고딕", 11))
    l.pack(anchor="w", padx=20)
    result_labels[lab] = l

# -------------------------------
# 버튼 기능
# -------------------------------
def start_recording():
    global running
    running = True
    threading.Thread(target=audio_loop, daemon=True).start()

def stop_recording():
    global running
    running = False

def calibrate():
    global threshold
    samples = []
    def callback(indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        samples.append(rms)
    with sd.InputStream(device=DEVICE_INDEX, callback=callback, channels=1, samplerate=samplerate):
        time.sleep(2)
    if samples:
        threshold = np.mean(samples) * 2
        rms_label.config(text=f"RMS: {samples[-1]:.5f} | threshold: {threshold:.5f}")

def list_devices():
    devices = sd.query_devices()
    print("=== 사용 가능한 입력 장치 목록 (JBL/Hands-Free 제외) ===")
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if dev['max_input_channels'] > 0:
            if "jbl" in name or "hands-free" in name:
                continue
            print(f"{i}: {dev['name']} (입력 {dev['max_input_channels']})")

tk.Button(btn_frame, text="시작", command=start_recording).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="일시정지", command=stop_recording).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="캘리브레이션(2초)", command=calibrate).grid(row=0, column=2, padx=5)
tk.Button(btn_frame, text="장치확인(콘솔)", command=list_devices).grid(row=0, column=3, padx=5)

# -------------------------------
# 오디오 루프
# -------------------------------
def audio_loop():
    global running
    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=1,
        samplerate=samplerate,
        blocksize=blocksize,
        dtype="float32"
    ) as stream:
        while running:
            data, _ = stream.read(blocksize)
            rms = np.sqrt(np.mean(data**2))

            if rms > threshold:
                status_label.config(text="수집 시작")
                audio = data[:samplerate].flatten()  # 2초 중 앞쪽 1초만 사용
                probs = predict_sound(audio)
                for i, lab in enumerate(labels):
                    percent = int(probs[i] * 100)
                    result_labels[lab].config(text=f"{lab}: {percent}%")
            else:
                status_label.config(text="소리 없음")
                for lab in labels:
                    result_labels[lab].config(text=f"{lab}: 0%")

            rms_label.config(text=f"RMS: {rms:.5f} | threshold: {threshold:.5f}")
            time.sleep(0.1)

root.mainloop()
