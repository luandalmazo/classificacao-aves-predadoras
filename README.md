# CLASSIFICADOR DE AVES PREDADORAS

Feito por:

- Luan Matheus Trindade Dalmazo
- Isadora Botassari

## Instalação de dependências

Instale as dependências necessárias (sugerimos a criação de um ambiente venv).
```pip install -r requirements.txt```

## Criação de arquivo .env

Lembre-se de criar um arquivo ***.env*** antes de rodar todos os códigos desse repositório, ele deverá ser da seguinte forma:

```
MAIN_DIR= DIRETÓRIONOQUALODATASETORIGINALESTÁ
TEST_DIR= NOVOCAMINHOPARAODATASET/TEST
VALIDATION_DIR=NOVOCAMINHOPARAODATASET/VALIDATION
TRAIN_DIR=NOVOCAMINHOPARAODATASET/TRAIN
AUGMENT=True -> CASO SEJA DESEJADO EXECUTAR O DATA AUGMENTATION
NUM_EPOCHS=20
```



## Descrição dos arquivos

Conforme descrito no *paper*, a *pipeline* consiste de diferentes etapas, abaixo todas elas são descritas (em ordem):

1. ***split_dataset.py*** 

Nesse arquivo dividimos o dataset **UTFPR-BOP: Birds of Prey** em 75% para treinamento, sendo 25% desse total destinado para validação. 25%  do *dataset* reservado para a etapa de teste.

2. ***preprocessing.py*** 

Aqui nós realizaremos o *data augmentation*, lembre-se de verificar se a variável de ambiente AUGMENT está configurada para TRUE.

3. ***train_families.py*** ou ***train_species.py***

O treinamento é realizado aqui.

4. ***test_families.py*** ou ***test_species.py***

O teste é realizado aqui, para o arquivo de espécies, é gerado uma matriz de confusão para acompanhar os acertos e erros do modelo.

5. ***classify.example.py***

Caso deseje classificar um exemplo individualmente por espécie, use este arquivo.

## DATASET

O *dataset* utilizado foi disponibilizado publicamente em [UTFPR-BOP Dataset](http://labic.utfpr.edu.br/datasets/UTFPR-BOP.html), todos os créditos das imagens utilizadas no treinamento, validação e teste são dadas ao laboratório [LABIC - UTFPR](http://labic.utfpr.edu.br/).