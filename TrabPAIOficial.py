

from logging import root
from multiprocessing.connection import wait
from scipy.spatial import distance as dist
from ntpath import realpath
import tkinter as tk
from tkinter import ttk
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn import svm, datasets
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import visualkeras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add, Dense, Input, GlobalAveragePooling2D
import xgboost as xgb
import numpy as np
import glob
import ctypes
import xgboost as xgb
import random 
import os
import time
import imutils
import timeit
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()



SZ=224
root = tk.Tk()
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
raios_x =  glob.glob("kneeKL224/train/0/*.png")
cortada = False
bin_n = 16
cantoSuperiorEsquerdo=[]
cantoInferiorDireito=[]

#Metodo para fazer corte de imagem com o mouse
def corte_com_mouse(mouse, x, y, flags, param):

    global cantoSuperiorEsquerdo, cantoInferiorDireito

# verifica quando o botão do mouse é pressionado e salva as coordenadas iniciais do corte
    if mouse == cv2.EVENT_LBUTTONDOWN:
        cantoSuperiorEsquerdo = [(x,y)]

# verifica quando o botão do mouse é levantado e salva as coordenadas finais do corte
    elif mouse == cv2.EVENT_LBUTTONUP:
        cantoInferiorDireito = [(x,y)]
        ponto_de_referencia = [cantoSuperiorEsquerdo[0], cantoInferiorDireito[0]]

# usa das coordenadas para cortar a imagem original, em seguida a imagem cortada é salva em um arquivo e projetada na tela
        corte_da_imagem = imagem_original [ponto_de_referencia[0][1]:ponto_de_referencia[1][1], ponto_de_referencia[0][0]:ponto_de_referencia[1][0]]
        cv2.imwrite('corte.png',corte_da_imagem)
        cv2.imshow("cortada", corte_da_imagem)

#Metodo para fazer correlação cruzada
def correlacao_cruzada():
    #a imagem original e seu corte sao retirados da tela e a imagem que sera usada para comparação é escolhida aleatoriamente
    cv2.destroyAllWindows()
    imagem_comparada = cv2.imread(random.choice(raios_x))
    corte = cv2.imread("corte.png")
    
    #escalas de cinza das imagens são convertidas para o mesmo espaço e a função de correlação cruzada é feita com base no corte da imagem original e uma imagem da base de dados escolhida aleatoriamente
    imagem_escala_cinza = cv2.cvtColor(imagem_comparada, cv2.COLOR_BGR2GRAY)
    corte_escala_cinza = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    resultado = cv2.matchTemplate(imagem_escala_cinza, corte_escala_cinza, cv2.TM_CCOEFF_NORMED)
    (valor_minimo, valor_maximo, localizacao_minima, localizacao_maxima) = cv2.minMaxLoc(resultado)
    
    #O tamanho do corte da imagem original é usado para determinar o tamanho que o retangulo indicador da região desejada será
    (valor_inicial_coordenada_x, valor_inicial_coordenada_y) = localizacao_maxima
    valor_final_coordenada_x = valor_inicial_coordenada_x + corte.shape[1]
    valor_final_coordenada_y = valor_inicial_coordenada_y + corte.shape[0]
    
    #um retangulo é criado sobre a imagem comparada e a imagem com o retangulo são exibidos
    cv2.rectangle(imagem_comparada, (valor_inicial_coordenada_x, valor_inicial_coordenada_y), (valor_final_coordenada_x, valor_final_coordenada_y), (255, 0, 0), 2)
    cv2.imshow("Imagem comparada", imagem_comparada)
    cv2.waitKey(0)

#Metodo referente a parte 1 do trabalho, cortar e comparar uma imagem usando correlação cruzada
def cortar():
    global imagem, imagem_comparada, imagem_original

    #Imagens com melhor resultado 
    #"D:/Trabalhos/trabalhos PAI/KneeXrayData/ClsKLData/kneeKL224/train/4\9900761R.png"
    #"D:/Trabalhos/trabalhos PAI/KneeXrayData/ClsKLData/kneeKL224/train/4\9171580R.png"

    # Seleção de imagens aleatorias a cada execução
    path = random.choice(raios_x)
    imagem = cv2.imread(path)
    #print(path)

    #copia da imagem original é criada para que a original possa ser cortada
    imagem_original = imagem.copy()
    inicio_coordenada_x, inicio_coordenada_y, fim_coordenada_x, fim_coordenada_y = 0, 0, 0, 0

    #apresenta a imagem original na tela e permite que o segmento desejado seja cortaco pela função corte_com_mouse
    cv2.namedWindow("Corte a imagem com o mouse")
    cv2.imshow("Corte a imagem com o mouse", imagem)
    cv2.setMouseCallback("Corte a imagem com o mouse", corte_com_mouse)

def EqualizarDuplicar():
    
    path = "imagensPreparadas/"

    # Caso o path descrito acima nao exista, ela sera criado
    if not os.path.exists(path):
        os.mkdir(path)

    # Percorre todas as pastas do diretoria descrito a baixo, cria pastas com o valor da categoria no diretorio descrito em path
    for i in range (5):
        imagens = glob.glob("C:\Trabalhos\PAI\data\kneeKL224/train/" +  str(i) + "/*.png")

        if not os.path.exists(path + str(i)):
            os.mkdir(path + str(i))

            # com o uso da funcao equalizeHist todas as imagens tem seu histograma equalizado
            for count, img_path in enumerate(imagens): 
                imagem = cv2.imread(img_path, 0)
                equ = cv2.equalizeHist(imagem)
                cv2.imwrite(path + str(i) + "/" + str(count) + ".jpg", equ)

            equalizadas = glob.glob (path + str(i) + "/" + "*")

            # pega as imagens equalizadas e cria copias espelhadas em suas repectivas categorias
            for contador, img_path in enumerate(equalizadas, start = count):  
                #print(img_path)
                imagem = cv2.imread(img_path, 0)
                img_flip_lr = cv2.flip(imagem, 1)
                cv2.imwrite(path + str(i) + "/" + str(contador) + ".jpg", img_flip_lr)


####################################################### testa o modelo

def classificadorSVM(seletor):
    # inicia a contagem do tempo
    tempo = timeit.timeit()
    
    ################################################################gerar data.pickle
    dir = 'C:\\Trabalhos\\PAI\\imagensPreparadas'
    # determina com base no tipo de execucao quais categorias devem ser selecionadas
    if seletor == 0:
        categorias = ['0', '1', '2', '3', '4']
    else:
        categorias = ['0','4']

    data = []

    # serializa todas as imagens das categorias determinadas acima em um vetor com a funcoes flatten que e em seguida colocado no vetor data 
    for categoria in categorias:
        path = os.path.join(dir, categoria)
        label = categorias.index(categoria)

        for img in os.listdir(path):
            imgPath = os.path.join(path,img)
            imagem = cv2.imread(imgPath, 0)
            vetImagem = np.array(imagem).flatten()

            data.append([vetImagem, label]) 

    # o vetor data compoe o arquivo .pickle, seu nome e conteudo variando em relacao ao tipo de funcao que e executada
    if seletor == 0:
        pick_in = open('data.pickle', 'wb')
    else:
        pick_in = open('dataBinario.pickle', 'wb')
    pickle.dump(data, pick_in)
    pick_in.close()

    # inicia a contagem do tempo
    tempo = timeit.timeit()

    # determina qual arquivo .pickle abrir variando em relacao ao tipo de funcao que e executada
    if seletor == 0:
        pick_in = open('data.pickle', 'rb')
    else:
        pick_in = open('dataBinario.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    # embaralha os dados e cria os vetores features e labels
    random.shuffle(data)
    features = []
    labels = []

    # relaciona as imagens e suas respectivas caracteristicas aos vetores ciados acima 
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    # a porcentagem de dados que serao usados para teste e treino e definida e os vetores features e labels sao 
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

    
    ####################################################### Gera o modelo
    # linha responsavel por criar o modelo  
    model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', shrinking = true).fit(xtrain, ytrain)

    # determina qual arquivo criar variando em relacao ao tipo de funcao que e executada
    if seletor == 0:
        pick = open('modelSVM.sav', 'wb')
    else:
        pick = open('modelSVMBinario.sav', 'wb')
    pickle.dump(model, pick)
    pick.close() 
    
    # determina qual arquivo abrir variando em relacao ao tipo de funcao que e executada
    if seletor == 0:
        pick = open('modelSVM.sav', 'rb')
    else:
        pick = open('modelSVMBinario.sav', 'rb')
    model = pickle.load(pick)
    pick.close() 

    # testa o modelo 
    prediction = model.predict(xtest)
    accuracy = model.score(xtest, ytest)

    # determina com base no tipo de execucao quais categorias devem ser selecionadas
    if seletor == 0:
        categorias = ['0', '1', '2', '3', '4']
    else:
        categorias = ['0','4']

    # apresenta resultados 
    print_matrix = ""
    if seletor == 0:
        results = multilabel_confusion_matrix(ytest, prediction)
        for count, confusion in enumerate(results):
            TP, TN, FP, FN = confusion.ravel()
            print_matrix = "Matrix of {:.0f}: True Positive: {:.0f} , True Negative: {:.0f}, False positive: {:.0f}, False Negative: {:.0f}".format(count, TP,TN, FP, FN)
            print(print_matrix)
    else:
        TP, TN, FP, FN = confusion_matrix(ytest, prediction , normalize='pred').ravel()
        print_matrix = "True Positive: {:.2%} , True Negative: {:.2%}, False positive: {:.2%}, False Negative: {:.2%}".format(TP,TN, FP, FN)
        print(print_matrix)
    print(skm.classification_report(ytest, prediction))
    ctypes.windll.user32.MessageBoxW(0, print_matrix, "Metrics", 1)
    ctypes.windll.user32.MessageBoxW(0, skm.classification_report(ytest, prediction), "Metrics", 1)

    print("acabou !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("tempo: ")
    print(tempo)

    imagem = xtest[0].reshape(224, 224)

    plt.imshow(imagem, cmap = 'gray')
    plt.show()


def classificadorXGBoost(seletor):
    # inicia a contagem do tempo
    tempo = timeit.timeit()
    ########################################################gera o modelo
    pick_in = ""
    if seletor == 0:
        pick_in = open('data.pickle', 'rb')
    else:
        pick_in = open('dataBinario.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

    model = xgb.XGBClassifier(learning_rate = 0.01,max_depth = 6,min_child_weight = 1,subsample = 0.7,gamma =0)

    model.fit(xtrain, ytrain)
    
    pick = ""
    if seletor == 0:
        pick = open('modelXGB.sav', 'wb')
    else:
        pick = open('modelXGBBinario.sav', 'wb')
    pickle.dump(model, pick)
    pick.close() 

    ####################################################### testa o modelo
    pick_in = ""
    if seletor == 0:
        pick_in = open('data.pickle', 'rb')
    else:
        pick_in = open('dataBinario.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

    pick = ""
    if seletor == 0:
        pick = open('modelXGB.sav', 'rb')
    else:
        pick = open('modelXGBBinario.sav', 'rb')
    model = pickle.load(pick)
    pick.close() 
    prediction = model.predict(xtest)
    accuracy = model.score(xtest, ytest)
    categorias = ['0','1','2','3','4']

    print('acuracia= ', accuracy )
    print('previsao= ', categorias[prediction[0]])

    imagem = xtest[0].reshape(224, 224)

    plt.imshow(imagem, cmap = 'gray')
    plt.show()

    
    print_matrix = ""
    if seletor == 0:
        results = multilabel_confusion_matrix(ytest, prediction)
        for count, confusion in enumerate(results):
            TP, TN, FP, FN = confusion.ravel()
            print_matrix = "Matrix of {:.0f}: True Positive: {:.0f} , True Negative: {:.0f}, False positive: {:.0f}, False Negative: {:.0f}".format(count, TP,TN, FP, FN)
            print(print_matrix)
    else:
        TP, TN, FP, FN = confusion_matrix(ytest, prediction , normalize='pred').ravel()
        print_matrix = "True Positive: {:.2%} , True Negative: {:.2%}, False positive: {:.2%}, False Negative: {:.2%}".format(TP,TN, FP, FN)
        print(print_matrix)
    print(skm.classification_report(ytest, prediction))
    ctypes.windll.user32.MessageBoxW(0, print_matrix, "Metrics", 1)
    ctypes.windll.user32.MessageBoxW(0, skm.classification_report(ytest, prediction), "Metrics", 1)
    print("acabou !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("tempo: ")
    print(tempo)

def classificadorMobilenetV2(seletor):
    """Method to generate and execute mobilenet v2"""
    def expansion_block(x, t, filters,block_id):
        """Expansion block"""
        prefix = 'block_{}_'.format(block_id)
        total_filters = t*filters
        x = Conv2D(total_filters,1,padding='same',use_bias=False, name =   prefix +'expand')(x)
        x = BatchNormalization(name=prefix +'expand_bn')(x)
        x = ReLU(6,name = prefix +'expand_relu')(x)    
        return x
    def depthwise_block(x, stride, block_id):    
        prefix = 'block_{}_'.format(block_id)
        x = DepthwiseConv2D(3, strides=(stride, stride), padding     ='same', use_bias=False, name=prefix + 'depthwise_conv')(x)
        x = BatchNormalization(name=prefix +'dw_bn')(x)
        x = ReLU(6, name=prefix +'dw_relu')(x)
        return x
    def projection_block(x, out_channels, block_id):
        prefix = 'block_{}_'.format(block_id)
        x = Conv2D(filters=out_channels, kernel_size=1, padding='same',    use_bias=False, name=prefix + 'compress')(x)
        x = BatchNormalization(name=prefix +'compress_bn')(x)
        return x
    def Bottleneck(x, t, filters, out_channels, stride,block_id):    
        y = expansion_block(x, t, filters, block_id)
        y = depthwise_block(y, stride, block_id)
        y = projection_block(y, out_channels, block_id)
        if y.shape[-1]==x.shape[-1]:
            y = add([x,y])
        return y
    def MobileNetV2(input_image, n_classes):
        input = Input(input_image)    
        x = Conv2D(32, kernel_size=3, strides=(2,2), padding='same', use_bias=False)(input)
        x = BatchNormalization(name='conv1_bn')(x)
        x = ReLU(6, name = 'conv1_relu')(x)    
        # 17 Bottlenecks    
        x = depthwise_block(x, stride=1, block_id=1)
        x = projection_block(x, out_channels=16, block_id=1)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)    
        x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)    
        #1*1 conv
        x = Conv2D(filters = 1280, kernel_size = 1, padding='same', use_bias=False, name = 'last_conv')(x)
        x = BatchNormalization(name='last_bn')(x)
        x = ReLU(6,name='last_relu')(x)   
        #AvgPool 7*7
        x = GlobalAveragePooling2D(name='global_average_pool')(x)    
        output = Dense(n_classes, activation='softmax')(x)    
        model = Model(input, output)    
        return model
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    train_data = ImageDataGenerator(rescale=1/255,
                                    rotation_range=15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    brightness_range=[0.5, 1.5],
                                    horizontal_flip=True,
                                    vertical_flip=True)
    test_data = ImageDataGenerator(rescale=1/255)
    train_data = 0
    test_data = 0
    if seletor == 0:
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(os.getcwd(), 'kneeKL224', 'train'),
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=16,
                shuffle=True        
        )
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(os.getcwd(), 'kneeKL224/', 'test'),
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=16,
                shuffle=True
        )
    else:
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(os.getcwd(), 'kneebinary/', 'train'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=16,
        shuffle=True        
        )
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(os.getcwd(), 'kneebinary/', 'test'),
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=16,
                shuffle=True
        )
    class_names = train_data.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    n_classes = 5
    model = MobileNetV2(input_shape, n_classes)
    model.summary()
    callback = [
        EarlyStopping(monitor='loss', patience=3, mode="auto"), 
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='min')
        ]
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer = "Adam",
                metrics=["accuracy"])


    checkpoint_filepath = 'modelito2/modelito'
    model.load_weights(checkpoint_filepath)
    checkpoint_filepath = 'modelito3/modelito'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(
            train_data,
            shuffle=True,
            validation_data=test_data,
            batch_size=16,
            verbose=1,
            epochs=1,
            callbacks=[model_checkpoint_callback ])
            
    checkpoint_filepath = ""
    if seletor == 0:
        checkpoint_filepath = 'modelMobileNetV2/modelito'
    else:
        checkpoint_filepath = 'modelMobileNetV2Binario/modelito'
    model.load_weights(checkpoint_filepath)
    model.evaluate(test_data, verbose=1)
    predictions = model.predict(test_data)
    predictions = np.argmax(predictions,axis=1)
    y_pred = []
    for i in predictions:
        if i >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
            plt.figure(figsize=(20, 20))
    for images, labels in test_data.take(1):
        for i in range(8):
            ax = plt.subplot(4, 2, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = model.predict(tf.expand_dims(images[i], 0))
            score = tf.nn.softmax(predictions[0])
            if(class_names[labels[i]]==class_names[np.argmax(score)]):
                plt.title("Actual: "+ class_names[labels[i]])
                plt.xlabel("Predicted: "+ class_names[np.argmax(score)],fontsize=14,fontdict={'color':'green'})
                
            else:
                plt.title("Actual: "+ class_names[labels[i]])
                plt.xlabel("Predicted: "+ class_names[np.argmax(score)],fontsize=14, fontdict={'color':'red'})
            plt.gca().axes.yaxis.set_ticklabels([])        
            plt.gca().axes.xaxis.set_ticklabels([])
    plt.show()

    #Predict
    y_prediction = model.predict(test_data)
    y_prediction = np.argmax (y_prediction, axis = 1)
    y_test=np.concatenate([y for x, y in test_data], axis=0)
    #Create confusion matrix and normalizes it over predicted (columns)
    results = multilabel_confusion_matrix(y_test, y_prediction)
    if seletor == 0:
        for count, confusion in enumerate(results):
            TP, TN, FP, FN = confusion.ravel()
            print_matrix = "Matrix of {:.0f}: True Positive: {:.0f} , True Negative: {:.0f}, False positive: {:.0f}, False Negative: {:.0f}".format(count, TP,TN, FP, FN)
            print(print_matrix)
            print_output = print_matrix + '\n'
            Accuracy = (TP+TN)/(TP+FP+FN+TN)
            accuracy_output = "Accuracy of {:.0f}: {:.2%}".format(count,Accuracy)
            print(accuracy_output)
            print_output = print_output + accuracy_output + '\n'
            Precision = TP/(TP+FP)
            precision_output = "Precision of {:.0f}: {:.2%}".format(count,Precision)
            print(precision_output)
            print_output = print_output + precision_output + '\n'
            Sensitivity = TP/(TP+FN)
            sensitivity_output = "Sensitivity of {:.0f}: {:.2%}".format(count,Sensitivity)
            print(sensitivity_output)
            print_output = print_output + sensitivity_output + '\n'
            F1_Score = 2*(Sensitivity * Precision) / (Sensitivity + Precision)
            f1_output = "F1 Score of {:.0f}: {:.2%}".format(count,F1_Score)
            print(f1_output)
            print_output = print_output + f1_output + '\n'
            Specificity = TN/(TN+FP)
            specificity_output = "Specifity of {:.0f}: {:.2%}".format(count,Specificity)
            print(specificity_output)
            print_output = print_output + specificity_output + '\n'
            ctypes.windll.user32.MessageBoxW(0, print_output, "Metrics", 1)

    else:
        TP, TN, FP, FN = confusion_matrix(y_test, y_prediction , normalize='pred').ravel()
        print_matrix = "True Positive: {:.2%} , True Negative: {:.2%}, False positive: {:.2%}, False Negative: {:.2%}".format(TP,TN, FP, FN)
        print(print_matrix)
        print_output = print_matrix + '\n'
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        accuracy_output = "Accuracy: {:.2%}".format(Accuracy)
        print(accuracy_output)
        print_output = print_output + accuracy_output + '\n'
        Precision = TP/(TP+FP)
        precision_output = "Precision: {:.2%}".format(Precision)
        print(precision_output)
        print_output = print_output + precision_output + '\n'
        Sensitivity = TP/(TP+FN)
        sensitivity_output = "Sensitivity: {:.2%}".format(Sensitivity)
        print(sensitivity_output)
        print_output = print_output + sensitivity_output + '\n'
        F1_Score = 2*(Sensitivity * Precision) / (Sensitivity + Precision)
        f1_output = "F1 Score: {:.2%}".format(F1_Score)
        print(f1_output)
        print_output = print_output + f1_output + '\n'
        Specificity = TN/(TN+FP)
        specificity_output = "Specifity: {:.2%}".format(Specificity)
        print(specificity_output)
        print_output = print_output + specificity_output + '\n'
        ctypes.windll.user32.MessageBoxW(0, print_output, "Confusion Matrix", 1)
        

root = tk.Tk()
canvas = tk.Canvas(root, height = 400, width= 300, bg = "#263D42")
canvas.pack()

def menuMobileNet():
    top = tk.Toplevel()
    top.title('Classificador MobileNetV2')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Classificador MobileNetV2 multiclasse", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorMobilenetV2(0))
    cortar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Classificador MobileNetV2 binario", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorMobilenetV2(1))
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.4)

    fechar_janela = tk.Button(top, text = "Retornar", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.1, relx = 0.1, rely = 0.7)

def menuSVM():
    top = tk.Toplevel()
    top.title('Classificador SVM')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Classificador SVM multiclasse", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorSVM(0))
    cortar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Classificador SVM binario", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorSVM(1))
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.4)

    fechar_janela = tk.Button(top, text = "Retornar", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.1, relx = 0.1, rely = 0.7)

def menuXGBoost():
    top = tk.Toplevel()
    top.title('Classificador XGBoost')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Classificador XGBoost multiclasse", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorXGBoost(0))
    cortar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Classificador XGBoost binario", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = lambda: classificadorXGBoost(1))
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.4)

    fechar_janela = tk.Button(top, text = "Retornar", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.1, relx = 0.1, rely = 0.7)

def selecionarClassificador():
    top = tk.Toplevel()
    top.title('Selecionar classificador')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Classificador Mobile", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = menuMobileNet)
    cortar_imagem.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Classificador SVM ", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = menuSVM)
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.3)

    correlacionar_imagem = tk.Button(top, text = "Classificador XGBoost ", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = menuXGBoost)
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.5)

    fechar_janela = tk.Button(top, text = "Fechar seletor", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.7)

def menuParte02():
    top = tk.Toplevel()
    top.title('Parte 02')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Selecionar classificador", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = selecionarClassificador)
    cortar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Equalizar e duplicar", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = EqualizarDuplicar)
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.4)

    fechar_janela = tk.Button(top, text = "Fechar Parte 02", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.7)

def menuParte01():
    top = tk.Toplevel()
    top.title('Parte 01')

    canvas = tk.Canvas(top, height = 400, width= 300, bg = "#263D42")
    canvas.pack()

    cortar_imagem = tk.Button(top, text = "Cortar imagem", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = cortar)
    cortar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.1)

    correlacionar_imagem = tk.Button(top, text = "Correlacao cruzada", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = correlacao_cruzada)
    correlacionar_imagem.place(relwidth = 0.8, relheight = 0.3, relx = 0.1, rely = 0.4)

    fechar_janela = tk.Button(top, text = "Fechar Parte 01", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = top.destroy)
    fechar_janela.place(relwidth = 0.8, relheight = 0.2, relx = 0.1, rely = 0.7)

parte01 = tk.Button(root, text = "Parte 01: Correlação cruzada", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = menuParte01)
parte01.place(relwidth = 0.8, relheight = 0.4, relx = 0.1, rely = 0.1)

parte02 = tk.Button(root, text = "Parte 02: Classificação", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = menuParte02)
parte02.place(relwidth = 0.8, relheight = 0.4, relx = 0.1, rely = 0.5)

root.mainloop()

