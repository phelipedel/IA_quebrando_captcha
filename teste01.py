import cv2                          #removcao de ruido
from PIL import Image               #trasformando tons

#Apenas para testar qual o melhor metodo para tratarmos a imagem


metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,

]

imagem = cv2.imread('bdcaptcha/1.png')

#transformar a image em escala de cinza

img_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

i = 0
for metodo in metodos: #para cada metodo vamos tratar a img
    i += 1
    _, imagem_tratata =  cv2.threshold(img_cinza, 127, 255, metodo or cv2.THRESH_OTSU)
    cv2.imwrite(f'teste_metodo/teste_metodo_{i}.png', imagem_tratata)
   #cv2.THRESH_OTSU tratamento auxiliar, melhor resultado
   # _, ignorar a primeira info, pois o thresh_outs me reotorna uma tubla com duas informacao
   # sendo uma inutil e outra util


imagem = Image.open('teste_metodo/teste_metodo_3.png')
imagem = imagem.convert('P')
imagem2 = Image.new('P', imagem.size, 255)

for x in range(imagem.size[1]): #tratamento de pixel X e Y , largura e coluna
    for y in range(imagem.size[0]):
        cor_pixel = imagem.getpixel((y, x))
        if cor_pixel < 115:
            imagem2.putpixel(( y , x ) ,  0 )

imagem2.save('teste_metodo/imagemfinal.png')