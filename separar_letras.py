import cv2
import os
import glob

arquivos = glob.glob('img_tratadas/*')
for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # PRETO E BRANCO
    _, nova_imagem = cv2.threshold(imagem, 0,255, cv2.THRESH_BINARY_INV)

    # encontrar os contornos de cada letra
    contornos,_ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL = contorno de imagem de fora para dentro
    # cv2.CHAIN_APPROX_SIMPLE = meto matematico

    regiao_letras = []
    # filtrar os contorno que sao realmente de letras
    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 115:
            regiao_letras.append((x,y,l,a))
            """
            l = largura
            a = altura 
            """
    # caso mais de 5 letrar, evitar que o sistema identifiquei riscos ou img indesejada
    if len(regiao_letras) != 5:
        continue

    # desenhar os contornos e separar as lestras em arquivos individuais

    imagem_final = cv2.merge([imagem] * 3)

    i = 0
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        imagem_letra = imagem[y-2:y+a+2, x-2:x+l+2] # para python primeiro Y depois X
        i += 1
        nome_arquivo = os.path.basename(arquivo).replace('.png', f'letra{i}.png')
        cv2.imwrite ( f'letras/{nome_arquivo}' , imagem_letra )
        cv2.rectangle(imagem_final, (x-2, y-2), (x+l, y+a), (0,255,255), 1 )
    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"identificar/{nome_arquivo}", imagem_final)