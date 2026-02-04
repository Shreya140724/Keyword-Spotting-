# 🎙️ Keyword Spotting Voice Command Application (TensorFlow Lite)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![TFLite](https://img.shields.io/badge/Model-TFLite-green)
![Audio](https://img.shields.io/badge/Input-Microphone-purple)
![GUI](https://img.shields.io/badge/UI-Tkinter-red)
![ML](https://img.shields.io/badge/Type-Keyword%20Spotting-success)
![Offline](https://img.shields.io/badge/Mode-Offline-important)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A lightweight **offline voice command recognition system** built using **TensorFlow Lite** and **Python Tkinter**.  
The application records audio from the microphone, converts it into a spectrogram, and predicts spoken keywords using a trained CNN model.

**Supported commands:**  
up, down, left, right, forward, backward, yes, no, stop, go, help

---

## 🚀 Features

- Offline speech command recognition  
- TensorFlow Lite inference  
- Tkinter GUI  
- Microphone recording using `sounddevice`  
- Spectrogram-based CNN classification  
- Confidence threshold filtering  
- Lightweight deployment model  

---

## 📂 Project Structure

Keyword Spotting/
│
├── GUI.py  
├── model.tflite  
├── background_noise.wav  
├── README.md  
└── Keyword.ipynb *(optional – training/experiments)*  

---

## ⚙️ Setup Instructions (Conda Recommended)

### 1️⃣ Create Environment

```bash
conda create -n keyword python=3.10
conda activate keyword
```

### 2️⃣ Install Dependencies

```bash
pip install numpy==1.26.4 tensorflow==2.12.0 keras==2.12.0
pip install sounddevice scipy wavio matplotlib seaborn pandas
```

### 3️⃣ Run Application

Ensure the following files are in the same directory:

```text
GUI.py
model.tflite
background_noise.wav
```

Run the application:

```bash
python GUI.py
```

---

## 🎤 Usage

1. Click **Record**  
2. Speak a single command clearly (e.g., *yes*, *up*, *stop*)  
3. Prediction appears in the GUI  
4. Confidence score is printed in the terminal  
5. Threshold can be tuned inside `GUI.py`  

---

## 🧠 Model Details

- **Input:** 1-second audio waveform (16 kHz)  
- **Feature:** Spectrogram (STFT)  
- **Architecture:** Lightweight CNN  
- **Output:** Softmax probabilities over 11 keywords  
- **Format:** TensorFlow Lite (`model.tflite`)  

---

## ⚠️ Notes

- Allow microphone access in **Windows Privacy Settings**  
- Best performance in a quiet environment  
- Speak **one word at a time**  
- Model is optimized for **demo / educational purposes**  

---

## 📌 Tech Stack

- Python 3.10  
- TensorFlow 2.12  
- TensorFlow Lite  
- NumPy / SciPy  
- SoundDevice  
- Tkinter  
- Matplotlib / Seaborn  

---

## 👩‍💻 Author

**Shreya Sidabache**
