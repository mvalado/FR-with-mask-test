import os
import cv2
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def load_images(parent_folder):
    # Crie listas vazias para armazenar as imagens e rótulos
    X = []
    y = []

    # Itere sobre as pastas e arquivos dentro da pasta-mãe usando os.walk
    for root, dirs, files in os.walk(parent_folder):
        # Itere sobre os arquivos
        for file_name in files:
            # Defina o caminho para o arquivo atual
            file_path = os.path.join(root, file_name)

            # Leia a imagem usando cv2.imread
            image = cv2.imread(file_path)

            # Redimensione a imagem para o tamanho desejado (opcional)
            image = cv2.resize(image, (224, 224))

            # Adicione a imagem à lista X
            X.append(image)

            # Adicione o rótulo (por exemplo, o nome da pessoa) à lista y
            y.append(os.path.basename(root))

    # Converta as listas X e y em matrizes NumPy
    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess_images(X, y):
    # Codifique os rótulos como inteiros
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Remova classes com menos de dois membros
    unique, counts = np.unique(y, return_counts=True)
    keep_classes = unique[counts >= 2]
    keep_indices = np.isin(y, keep_classes)
    X = X[keep_indices]
    y = y[keep_indices]

    # Divida os dados em conjuntos de treinamento e teste com estratificação e um tamanho maior para o conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # Normalize as imagens
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    return X_train, X_test, y_train, y_test

def extract_features(X_train, X_test):
    # Carregue a rede VGG16 pré-treinada sem a camada totalmente conectada superior
    base_model = VGG16(weights='imagenet', include_top=False)

    # Extraia características das imagens usando a rede VGG16 pré-treinada
    train_features = base_model.predict(X_train)
    test_features = base_model.predict(X_test)

    # Achate as características extraídas para que possam ser usadas como entrada para um classificador
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)

    return train_features, test_features

def create_database(train_features, y_train):
    # Conecte-se ao banco de dados SQLite3 (crie o banco de dados se ele não existir)
    conn = sqlite3.connect('endereço_para_o_banco_de_dados_(nome do banco).db')
    c = conn.cursor()

    # Crie uma tabela para armazenar os vetores descritores e os rótulos correspondentes
    c.execute('''
        CREATE TABLE IF NOT EXISTS descriptors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label INTEGER NOT NULL,
            descriptor BLOB NOT NULL
        )
    ''')

    # Insira os vetores descritores e os rótulos correspondentes na tabela (converta os vetores descritores em bytes usando tobytes)
    data = list(zip(y_train, [descriptor.tobytes() for descriptor in train_features]))
    c.executemany('INSERT INTO descriptors (label, descriptor) VALUES (?, ?)', data)

    # Confirme as alterações e feche a conexão com o banco de dados
    conn.commit()
    conn.close()

def train_classifier(train_features, y_train):
    # Treine um classificador nas características extraídas (por exemplo, um classificador SVM)
    clf = SVC(probability=True)
    clf.fit(train_features, y_train)

    return clf

def evaluate_classifier(clf, test_features, y_test):
    # Avalie o classificador no conjunto de teste
    accuracy = clf.score(test_features, y_test)
    print(f'Test accuracy: {accuracy:.2f}')

def add_person_to_database(image_path, label, base_model):
    # Carregue a imagem da pessoa sem máscara facial e pré-processe-a (por exemplo, redimensionando-a e normalizando-a)
    unmasked_person_image = cv2.imread(image_path)
    unmasked_person_image_processed = cv2.resize(unmasked_person_image, (224, 224))
    unmasked_person_image_processed = preprocess_input(unmasked_person_image_processed)

    # Calcule o vetor descritor da imagem da pessoa sem máscara facial
    unmasked_person_descriptor = base_model.predict(np.expand_dims(unmasked_person_image_processed, axis=0)).reshape(1, -1)

    # Conecte-se ao banco de dados SQLite3
    conn = sqlite3.connect('endereço_para_o_banco_de_dados_(nome do banco).db')
    c = conn.cursor()

    # Adicione o vetor descritor e o rótulo correspondente ao banco de dados
    data = (label, unmasked_person_descriptor.tobytes())
    c.execute('INSERT INTO descriptors (label, descriptor) VALUES (?, ?)', data)

    # Confirme as alterações e feche a conexão com o banco de dados
    conn.commit()
    conn.close()

def recognize_face(image_path, base_model):
    # Carregue a imagem da pessoa usando máscara facial e pré-processe-a (por exemplo, redimensionando-a e normalizando-a)
    masked_person_image = cv2.imread(image_path)
    masked_person_image_processed = cv2.resize(masked_person_image, (224, 224))
    masked_person_image_processed = preprocess_input(masked_person_image_processed)

    # Calcule o vetor descritor da imagem da pessoa usando máscara facial
    masked_person_descriptor = base_model.predict(np.expand_dims(masked_person_image_processed, axis=0)).reshape(1, -1)

    # Conecte-se ao banco de dados SQLite3
    conn = sqlite3.connect('endereço_para_o_banco_de_dados_(nome do banco).db')
    c = conn.cursor()

    # Recupere os vetores descritores e os rótulos correspondentes do banco de dados (converta os vetores descritores de volta em matrizes NumPy usando frombuffer)
    c.execute('SELECT label, descriptor FROM descriptors')
    data = c.fetchall()
    labels, descriptors = zip(*data)
    labels = np.array(labels)
    descriptors = np.vstack([np.frombuffer(descriptor, dtype=np.float32) for descriptor in descriptors])

    # Feche a conexão com o banco de dados
    conn.close()

    # Encontre o vetor descritor mais próximo do vetor descritor da pessoa usando máscara facial
    best_match_index = np.argmin(np.linalg.norm(descriptors - masked_person_descriptor, axis=1))
    best_match_label = labels[best_match_index]

    # Mostre o resultado do reconhecimento facial
    print(f'Best match: {label_encoder.inverse_transform([best_match_label])[0]}')

    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.show()

# Defina o caminho para a pasta-mãe
parent_folder = 'endereço_para_o_diretório_da_pasta_mãe'

# Carregue as imagens da pasta-mãe
X, y = load_images(parent_folder)

# Pré-processe as imagens
X_train, X_test, y_train, y_test = preprocess_images(X, y)

# Extraia características das imagens usando a rede VGG16 pré-treinada
train_features, test_features = extract_features(X_train, X_test)

# Crie um banco de dados para armazenar os vetores descritores e os rótulos correspondentes
create_database(train_features, y_train)

# Treine um classificador nas características extraídas
clf = train_classifier(train_features, y_train)

# Avalie o classificador no conjunto de teste
evaluate_classifier(clf, test_features, y_test)

# Adicione uma nova pessoa ao banco de dados
image_path = 'endereço_para_a_imagem_da_nova_pessoa_sem_máscara'
label = 'marcelinho'
add_person_to_database(image_path, label)

# Reconheça a face de uma pessoa usando máscara facial
image_path = 'endereço_para_a_imagem_da_nova_pessoa_usando_máscara'
recognize_face(image_path)