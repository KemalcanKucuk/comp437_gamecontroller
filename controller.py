import time
from pygame.locals import *
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import speech_recognition_tf_tutorial as srtf
import pyaudio
import wave

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

def record_command():
    # constant values are taken from the notebook
    chunk = 3200
    sample_format = pyaudio.paInt16
    channels = 1
    seconds = 1
    rate = 16000
    p = pyaudio.PyAudio()
    print("Start recording")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finish recording")
    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)

def predict_command():
    model = tf.keras.models.load_model('./saved_model.h5')
    audio = record_command()
    spectrogram = srtf.preprocess_audiobuffer(audio)
    pred = model(spectrogram)
    label_pred = np.argmax(pred, axis = 1)
    command = commands[label_pred[0]]
    print('Predicted label: ', command)

    return command

predict_command()







# EXPERIMENTATION + legacy from notebook

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def preprocess_audiobuffer(waveform):
    waveform =  waveform / 32768

    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    spectogram = get_spectrogram(waveform)
    spectogram = tf.expand_dims(spectogram, 0)
    
    return spectogram


"""
def command_recognition_sr(rec, mic):

    if not isinstance(rec, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(mic, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")
    
    print("Recording has started")
    with mic as source:
        rec.adjust_for_ambient_noise(source)
        audio = rec.listen(source)

    result = None

    # try to recognize, return the error if recognition is not possible
    try:
        result = rec.recognize_google(audio)
    except sr.RequestError:
        result = "API Unavailable!"
    except sr.UnknownValueError:
        result = "Speech unrecognizable"

    print(result)
    return result

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def command_execution_sr(speech):
    words = speech.split(" ")
    for word in words:
        if word.lower() == "left":
            newevent = pygame.event.Event(pygame.locals.KEYDOWN,  key=pygame.locals.K_LEFT, mod=pygame.locals.KMOD_NONE) #create the event
            pygame.event.post(newevent) #add the event to the queue
            print("bazort")
            break
        if word.lower() == "right":
            pass
            break
        if word.lower() == "jump":
            pass
            break
        if word.lower() == "down":
            pass
            break
        if word.lower() == "up":
            pass
            break
        if word.lower() == "shoot":
            pass
            break
    pass
    

#speech = command_recognition(recognizer, microphone)
#command_execution(speech)

"""

