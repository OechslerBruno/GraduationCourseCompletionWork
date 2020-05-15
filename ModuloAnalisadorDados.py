'''
Modulo Analisador de Dados, irá receber registro, proveniente do MPA, para classifica-lo.
    Fará a leitura do banco de dados, treinará o algoritmo e fará a predição do registro.
Data da criação: 20/10/2018
'''

import pandas as pd
import numpy  as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing, model_selection


class MAD():

    def analisaDador(self, registro):
        print("veio no analisa dados A")
        print("registro: ")
        print(registro)

        df = pd.read_csv('baseDadosTreinamentoMaisNovaMenosReg-1710-C.txt')  # leitura da base
        print("leu base")
        X = np.array(df.drop(['condicao'], 1))
        y = np.array(df['condicao'])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        print("fez o train")
        clf = LinearSVC(random_state=0, dual=True)
        clf.fit(X_train, y_train)

        print("fit")
        registroNp = np.array(registro)
        registroNp = registroNp.reshape(1, -1)
        print("reshape")

        ##print(clf.predict(registro))
        return clf.predict(registroNp)

