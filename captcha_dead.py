import cv2
import os
import glob
from PIL import Image

def tratar_img(pasta_origem, pasta_destinno = 'img_tratadas'):
        arquivos = glob.glob(f'{pasta_origem}/*')
        for arquivo  in arquivos:
            imagem = cv2.imread ( arquivo )
            # transformar a image em escala de cinza


            img_cinza = cv2.cvtColor ( imagem , cv2.COLOR_RGB2GRAY )
            _ , imagem_tratata = cv2.threshold ( img_cinza , 127 , 255 , cv2.THRESH_TRUNC or cv2.THRESH_OTSU )
            nome_arquivo = os.path.basename(arquivo) #usando o mesmo nome
            cv2.imwrite ( f'{pasta_destinno}/{nome_arquivo}' , imagem_tratata )

            # SEGUNDO TRATAMENTO -----------------------------

        arquivos = glob.glob(f'{pasta_destinno}/*')
        for arquivo in arquivos:
            imagem = Image.open ( arquivo )
            imagem = imagem.convert ( 'P' )
            imagem2 = Image.new ( 'P' , imagem.size , 255 )

            for x in range ( imagem.size [ 1 ] ) :  # tratamento de pixel X e Y , largura e coluna
                for y in range ( imagem.size [ 0 ] ) :
                    cor_pixel = imagem.getpixel ( (y , x) )
                    if cor_pixel < 115 :
                        imagem2.putpixel ( (y , x) , 0 )
            nome_arquivo = os.path.basename(arquivo)
            imagem2.save ( f'{pasta_destinno}/{nome_arquivo}' )



if __name__ == '__main__':
    tratar_img('bdcaptcha')