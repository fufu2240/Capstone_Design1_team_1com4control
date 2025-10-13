import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import time
import librosa
from tensorflow.keras.models import load_model
from tkinter import ttk

# -------------------------------
# 모델 불러오기
# -------------------------------
model = load_model(r"C:\capston\sound_event_model_fixed.h5")

labels = [
    "cry", "cough", "fall", "knock", "noise",
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

samplerate = 22050
duration = 1
blocksize = samplerate * duration
running = False
threshold = 0.01
DEVICE_INDEX = None

# -------------------------------
# 유지 로직 변수
# -------------------------------
last_probs = None
last_update_time = 0
HOLD_TIME = 2.0  # 최소 유지 시간(초)

# -------------------------------
# 오디오 전처리
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
    mel_db = mel_db[..., np.newaxis]
    mel_db = mel_db[np.newaxis, ...]
    return mel_db

def predict_sound(audio):
    global last_probs, last_update_time
    x = preprocess_audio(audio)
    pred = model.predict(x, verbose=0)[0]

    max_idx = np.argmax(pred)
    max_val = pred[max_idx]

    if max_val > 0.3:
        last_probs = pred
        last_update_time = time.time()

    if last_probs is not None and (time.time() - last_update_time) < HOLD_TIME:
        return last_probs
    else:
        return pred

# -------------------------------
# 기본 입력 장치 자동 탐색
# -------------------------------
def auto_detect_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if dev['max_input_channels'] > 0:
            if "jbl" in name or "hands-free" in name:
                continue
            print(f"🎯 자동 선택된 입력 장치: {i} - {dev['name']}")
            return i
    raise RuntimeError("❌ 유효한 입력 장치를 찾지 못했습니다.")

DEVICE_INDEX = auto_detect_device()

# -------------------------------
# Tkinter GUI
# -------------------------------
root = tk.Tk()
root.title("실시간 사운드 분류 (막대그래프 표시)")

title = tk.Label(root, text="실시간 사운드 인식", font=("맑은 고딕", 18, "bold"))
title.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

status_label = tk.Label(root, text="수집 대기", font=("맑은 고딕", 12))
status_label.pack(pady=5)

rms_label = tk.Label(root, text=f"RMS: 0.00000 | threshold: {threshold:.5f}", font=("맑은 고딕", 10))
rms_label.pack()

# -------------------------------
# 결과 표시 (텍스트 + 막대그래프)
# -------------------------------
result_widgets = {}
for lab in labels:
    frame = tk.Frame(root)
    frame.pack(anchor="w", padx=20)

    lbl = tk.Label(frame, text=f"{lab}: 0%", font=("맑은 고딕", 11), width=20, anchor="w")
    lbl.pack(side="left")

    canvas = tk.Canvas(frame, width=200, height=15, bg="white")
    canvas.pack(side="left", padx=5)

    result_widgets[lab] = (lbl, canvas)

# -------------------------------
# 장치 선택 드롭다운
# -------------------------------
devices = sd.query_devices()
input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]

device_var = tk.StringVar()
device_combo = ttk.Combobox(root, textvariable=device_var, values=input_devices, state="readonly", width=60)
device_combo.set(f"{DEVICE_INDEX}: {devices[DEVICE_INDEX]['name']} (기본)")
device_combo.pack(pady=5)

def change_device(event):
    global DEVICE_INDEX, running
    selected = device_combo.get()
    DEVICE_INDEX = int(selected.split(":")[0])
    print(f"🔄 장치 변경됨: {DEVICE_INDEX} - {devices[DEVICE_INDEX]['name']}")
    if running:
        stop_recording()
        start_recording()

device_combo.bind("<<ComboboxSelected>>", change_device)

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

tk.Button(btn_frame, text="시작", command=start_recording).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="일시정지", command=stop_recording).grid(row=0, column=1, padx=5)

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
                audio = data.flatten()
                probs = predict_sound(audio)

                for i, lab in enumerate(labels):
                    percent = int(probs[i] * 100)
                    lbl, canvas = result_widgets[lab]
                    lbl.config(text=f"{lab}: {percent}%")

                    # 막대그래프 갱신
                    canvas.delete("bar")
                    bar_len = int(2 * percent)  # 최대 200px
                    if bar_len > 0:
                        canvas.create_rectangle(0, 0, bar_len, 15, fill="blue", tags="bar")

            else:
                status_label.config(text="소리 없음")
                for lab in labels:
                    lbl, canvas = result_widgets[lab]
                    lbl.config(text=f"{lab}: 0%")
                    canvas.delete("bar")

            rms_label.config(text=f"RMS: {rms:.5f} | threshold: {threshold:.5f}")
            time.sleep(0.1)

root.mainloop()
