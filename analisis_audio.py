import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import matplotlib
matplotlib.use('Agg')


def audio():
    audio, sr = librosa.load('song_1.wav')

    return audio, sr


audio_data, sr = audio()

print("Audio:", audio_data)
print("Tasa de muestreo:", sr)


def feat(audio, sr):
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo, beats


audio_data, sr = audio()
tempo, beats = feat(audio_data, sr)

print("Tempo", tempo)
print("Beats", beats)

plt.plot(audio_data)
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Audio en el dominio del tiempo')
plt.savefig('audio_plot.png')

sd.play(audio_data, sr)
sd.wait()
