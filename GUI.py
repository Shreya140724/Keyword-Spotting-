import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# Set the TensorFlow verbosity level
tf.get_logger().setLevel('ERROR')

freq = 16000

# Commands: ['help' 'up' 'right' 'yes' 'left' 'no' 'stop' 'down' 'backward' 'forward' 'go']
#commands = ['help', 'up', 'right', 'yes', 'left', 'no', 'stop', 'down', 'backward', 'forward', 'go']
commands = ['up', 'right', 'help', 'backward', 'down', 'forward', 'yes', 'go', 'no', 'left', 'stop']
num_labels = len(commands)
threshold = 0.25

# Decode audio binary data
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

# Get spectrogram from audio waveform
def get_spectrogram(waveform):
    input_len = freq
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([freq] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Record audio and save to a file
def record():
    duration = 2
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()
    file_path = "background_noise.wav"
    wv.write(file_path, recording, freq, sampwidth=2)
    return file_path

# Run TensorFlow Lite model inference
def run_tflite(tflite_model_path, spectrogram):
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], spectrogram)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    max_index = np.argmax(output)
    print(f"Confidence: {output[max_index]:.2f}")
    # return commands[max_index]

    if output[max_index] >= threshold:
        return commands[max_index]
    else:
        return "background Noise"

# Perform voice command prediction
def predict():
    file_path = record()
    # file_path = "/home/sagarkhimani/Downloads/Word_GunShot/data/yes/6f3458b3_nohash_1.wav"
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    spectro = get_spectrogram(waveform)
    spectro_numpy = spectro.numpy()
    spectro_expanded = np.expand_dims(spectro_numpy, axis=0)
    tflite_model_path = "model.tflite"
    out = run_tflite(tflite_model_path, spectro_expanded)
    pred_lbl.config(text=f"  {out}  ")

if __name__ == '__main__':
    master = tk.Tk()
    master.geometry('500x300')
    master.title("Voice Commands")

    window = tk.Frame(master, width=100, height=100, highlightbackground="black", highlightthickness=2)
    window.place(in_=master, anchor="c", relx=.5, rely=.5)

    Label(window, text="Voice Recorder", font='Helvetica 14 bold').grid(row=2, column=1, padx=20, pady=10)
    lbl1 = Label(window, text="Prediction", font='Helvetica 16 bold')
    lbl1.grid(row=4, column=1, padx=5, pady=10)

    progress = Progressbar(window, orient=HORIZONTAL, length=100, mode='determinate')

    def bar():
        import time
        progress['value'] = 50
        master.update_idletasks()
        time.sleep(1)
        progress['value'] = 100
        master.update_idletasks()
        time.sleep(1)
        progress['value'] = 0

    progress.grid(row=3, column=2, padx=10, pady=10)

    pred_lbl = tk.Label(window, text="  Result  ", fg='Green', font='Helvetica 14 bold', borderwidth=2, relief="solid")
    pred_lbl.grid(row=4, column=2, padx=10, pady=10)

    b = tk.Button(window, text="Record", command=predict)
    b.configure(font=('Sans', '14', 'bold'), background='Yellow')
    b.grid(row=3, column=1, padx=5, pady=10)

    master.mainloop()
