import os


# Sou preguicoso mesmo que mane ficar criando pastinha de letra rs :)
while True:
    while True:
        folder = input('Nome da pasta: ')
        try:
            directory = folder
            parent_dir = 'base_letras'
            path = os.path.join ( parent_dir , directory )
            os.mkdir ( path )
        except (FileExistsError ,) :
            directory = os.path.exists ( 'base_letras' )
            print ( f'Erro! Diretorio ja existe,  tente novamente.  ' )
        else :

            print ( f'Gerando pasta...' )
            break
