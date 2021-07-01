import cv2
import pickle
import numpy as np
from imutils import paths

from BODYBUILDER_MODELO import modelo
from helpers import resize_to_fit
from keras.models import load_model
from captcha_dead import tratar_img


def brut_captcha () :
    with open ( 'rotulos_modelo.dat' , 'rb' ) as arquivo_tradutor :
        lb = pickle.load ( arquivo_tradutor )

    modelo = load_model ( "modelo_treinado.hdf5" )

    # usando modelo
    tratar_img ( 'resolver' , pasta_destinno = 'teste_aplicada_resolvendo' )

    # ler todos os arquivos da pasta 'resolver'
    arquivos = list ( paths.list_images ( 'teste_aplicado_resolvendo' ) )
    for arquivo in arquivos :
        imagem = cv2.imread ( arquivo )
        imagem = cv2.cvtColor ( imagem , cv2.COLOR_RGB2GRAY )
        # PRETO E BRANCO
        _ , nova_imagem = cv2.threshold ( imagem , 0 , 255 , cv2.THRESH_BINARY_INV )

        # encontrar os contornos de cada letra
        contornos , _ = cv2.findContours ( nova_imagem , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
        # cv2.RETR_EXTERNAL = contorno de imagem de fora para dentro
        # cv2.CHAIN_APPROX_SIMPLE = meto matematico

    regiao_letras = [ ]
    # filtrar os contorno que sao realmente de letras
    for contorno in contornos :
        (x , y , l , a) = cv2.boundingRect ( contorno )
        area = cv2.contourArea ( contorno )
        if area > 115 :
            regiao_letras.append ( (x , y , l , a) )
            """
            l = largura
            a = altura 
            """

    # desenhar os contornos e separar as lestras em arquivos individuais

    regiao_letras = sorted ( regiao_letras , key = lambda x : x [ 0 ] )  # ordenando as letras de acordo com a img
    # desenhar os contornos e separar as letras em arquivos individuais
    imagem_final = cv2.merge ( [ imagem ] * 3 )
    previsao = [ ]

    i = 0
    for retangulo in regiao_letras :
        x , y , l , a = retangulo
        imagem_letra = imagem [ y - 2 :y + a + 2 , x - 2 :x + l + 2 ]  # para python primeiro Y depois X

        imagem_letra = resize_to_fit ( imagem_letra , 20 , 20 )

        imagem_letra = np.expand_dims ( imagem_letra , axis = 2 )  # Adicionando a 4 dimensao
        imagem_letra = np.expand_dims ( imagem_letra , axis = 0 )

        letra_prevista = modelo.predict ( imagem_letra )
        letra_prevista = lb.inverse_transform ( letra_prevista ) [ 0 ]
        previsao.append ( letra_prevista )

    texto_previsao = "".join ( previsao )
    print ( texto_previsao )
    return texto_previsao  # caso queria mais de uma respota deixa desativo


if __name__ == '__main__' :
    brut_captcha ( )
