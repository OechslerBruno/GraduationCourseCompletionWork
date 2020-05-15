'''
Modulo Processador de Audio, destinado a realizar a leitura e extração de dados do arquivo de áudio recebido.
    Irá retornar um registro composto pelos dados extraídos.
Data da criação: 20/10/2018
'''

#Imports para modulo MPA
import librosa
import numpy as np
from numpy import linalg as LA


class MPA():

    def procAudio(self, camArq):

        print("Veio proc audio no MPA")
        # registro = np.array()

        print("camArq: " + camArq)
        audio, duracao = librosa.load(camArq)  ##le o arquivo de audio

        print("MFCC")
        # MFCC
        mfcc = librosa.feature.mfcc(audio)
        valorMFCC = LA.norm(mfcc)
        print(valorMFCC)
        #registro.append(valorMFCC)

        print("Polynomial")
        # Polynomial features
        poly = librosa.feature.poly_features(audio)
        valorPoly = LA.norm(poly)
        #registro.append(valorPoly)

        print("Tonnetz")
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(audio)
        valorTonnetz = LA.norm(tonnetz)
        #registro.append(valorTonnetz)

        print("melSpectrogram")
        # melSpectrogram
        melSpectrogram = librosa.feature.melspectrogram(audio)
        valorMelSpect = LA.norm(melSpectrogram)
        #registro.append(valorMelSpect)

        print("magnitude")
        # magnitude
        magnitude = librosa.core.magphase(audio)
        valorMagnitude = LA.norm(magnitude)
        #registro.append(valorMagnitude)

        print("pass magnitude")

        #print(np.asarray(registro))
        ##registro = np.array([valorMFCC, valorPoly, valorTonnetz, valorMelSpect, valorMagnitude])
        registro = [valorMFCC, valorPoly, valorTonnetz, valorMelSpect, valorMagnitude]
        print(registro)
        return registro


#print(MPA().procAudio("audio_treinamento_novo/murmur__112_1306243000964_B.wav"))