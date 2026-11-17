# SLICAI

Este repositório contém o código do SLICAI, protótipo desenvolvido para o meu TCC. A aplicação recebe uma imagem dermatoscópica e apresenta uma classificação preliminar da lesão, a probabilidade estimada de malignidade e uma máscara com o contorno previsto. As análises também podem ser consultadas posteriormente pela interface web.

O sistema é experimental e não fornece diagnóstico médico.

## Organização dos arquivos

- `api/`: API, conexão com o banco e armazenamento dos registros.
- `src/`: scripts de preparação dos dados, treinamento, avaliação e análise de imagens.
- `web/`: interface da aplicação.
- `models/`: modelos finais utilizados pela aplicação.
- `reports/`: métricas e figuras geradas nos experimentos.

Os datasets e as imagens enviadas pela aplicação não fazem parte do repositório.

## Modelos e dados

O classificador foi treinado com imagens do HAM10000 e avaliado em um conjunto de teste interno e no BCN20000, usado como teste externo. O segmentador foi treinado com imagens e máscaras da ISIC 2016.

Para executar a aplicação, os seguintes arquivos devem estar em `models/`:

- `classifier_efficientnet_b0_selected.pth`
- `segmenter_unet_isic2016.pth`

## Instalação das dependências

Na raiz do projeto, crie o ambiente virtual e instale as dependências:

powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt


## Banco de dados

A aplicação utiliza PostgreSQL. Crie um banco para o projeto, copie `.env.example` para `.env` e preencha as variáveis de conexão com os dados da sua instalação. O arquivo `.env` não deve ser enviado ao Git.

Na raiz do projeto, crie as tabelas com:

powershell
.\.venv\Scripts\python.exe -m api.create_tables


## Executar a aplicação

Após instalar as dependências e configurar o banco, execute:

powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload

A interface estará disponível em `http://127.0.0.1:8000`.

## Uso previsto

O SLICAI foi desenvolvido para fins acadêmicos e de apoio preliminar à análise de imagens. Seus resultados não substituem avaliação clínica, diagnóstico médico ou exames complementares.