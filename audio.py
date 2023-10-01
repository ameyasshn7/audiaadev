import numpy as np
import librosa as lr
import librosa.display
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import soundfile as sf
import os
class processing():
    def __init__(self, file_name):
        """
        Initialize the audio processing class with the given audio file.
        """
        self.audio_file, self.sfreq = lr.load(file_name)
        self.n_fft = 2048
        self.hop_length = 512
        self.ft = np.abs(librosa.stft(self.audio_file[:self.n_fft], hop_length=self.hop_length))
        self.graph_path = 'E:/school/college/masters/Projects/audapp2/graphs'

    @staticmethod
    def allowed_file(filename):
        """
        Check if the provided filename has an allowed extension.
        """
        allowed_extensions = {'mp3', 'wav'}  # Add allowed file extensions here
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    def process_audio(self):
        """
        Process the audio file, calculate integrated loudness, and return the result.
        """
        # time = np.arange(0, len(self.audio_file)) / self.sfreq
        plt.figure(figsize=(20, 5))
        librosa.display.waveshow(self.audio_file, sr=self.sfreq)
        meter = pyln.Meter(self.sfreq)
        loudness = meter.integrated_loudness(self.audio_file)
        return loudness


    def fft(self):
        """
        Compute the FFT (Fast Fourier Transform) of the audio file and return magnitude and frequency components.
        """
        ft = self.ft
        fft_analysis = np.fft.fft(self.audio_file)
        magnitude = np.abs(fft_analysis)
        frequency = np.linspace(0, self.sfreq, len(magnitude))
        left_mag = magnitude[:len(magnitude) // 2]
        left_freq = frequency[:len(frequency) // 2]
        return left_mag, left_freq

    def spectrogram(self):
        """
        Compute and display the spectrogram of the audio file.
        """
        hop_length = 512
        ft = np.abs(librosa.stft(self.audio_file, n_fft=self.n_fft, hop_length=512))
        librosa.display.specshow(ft, sr=self.sfreq, x_axis='time', y_axis='linear')
        plt.colorbar()
        ft_dB = librosa.amplitude_to_db(ft, ref=np.max)
        # librosa.display.specshow(ft_dB, sr=self.sfreq, hop_length=512, x_axis='time', y_axis='log')
        spectrogram = os.path.join(self.graph_path,'spectrogram.png')
        plt.savefig(self.graph_path)
        plt.close
        return spectrogram

    def stft(self):
        """
        Compute and return the short-term Fourier transform (STFT) of the audio file.
        """
        stft_seg = librosa.feature.melspectrogram(y=self.audio_file, hop_length=self.hop_length, n_fft=self.n_fft)
        spectrogram = np.abs(stft_seg)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        # stft_plot = plt.figure(figsize=(20,5))
        stft_plot = os.path.join(self.graph_path,'stft.png')
        
        plt.savefig(self.graph_path)
        plt.close()
        return stft_plot
        # return stft_plot.show

    def getMelSpec(self):
        """
        Compute and display the Mel spectrogram of the audio file.
        """
        mel_signal = librosa.feature.melspectrogram(y=self.audio_file, sr=self.sfreq, hop_length=self.hop_length,
                                                   n_fft=self.n_fft)
        spectrogram = np.abs(mel_signal)
        power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
        plt.figure(figsize=(8, 7))
        librosa.display.specshow(power_to_db, sr=self.sfreq, x_axis='time', y_axis='mel', cmap='magma',
                                 hop_length=self.hop_length)
        plt.colorbar(label='dB')
        plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
        plt.xlabel('Time', fontdict=dict(size=15))
        plt.ylabel('Frequency', fontdict=dict(size=15))
        # melspec = plt.savefig(self.graph_path)
        melspec_plot = os.path.join(self.graph_path,'melSpec.png')
        plt.savefig(self.graph_path) 
        plt.close()
        return melspec_plot
        # return plt.show()

    def getChroma(self):
        """
        Compute and display chroma features of the audio file.
        """
        chroma = librosa.feature.chroma_stft(S=self.ft, sr=self.sfreq)
        chroma = np.cumsum(chroma)
        x = np.linspace(-chroma, chroma)
        plt.plot(x, np.sin(x))
        plt.xlabel('Angle [rad]')
        plt.ylabel('sin(x)')
        plt.axis('tight')
        chroma_plot = os.path.join(self.graph_path,'chroma.png')

        plt.savefig(self.graph_path)
        plt.close
        return chroma_plot
        # return plt.show()
