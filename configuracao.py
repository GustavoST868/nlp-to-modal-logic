import torch

class Configuracao:
    # paths/caminhos
    CAMINHO_CONJUNTO_DADOS = "dataset.json"
    NOME_MODELO = "google/mt5-small"  # base model for multilingual seq2seq/modelo base para seq2seq multilíngue
    DIRETORIO_SAIDA = "./modelo_final"
    
    # training hyperparameters/hiperparâmetros de treinamento
    COMPRIMENTO_MAXIMO_ENTRADA = 64
    COMPRIMENTO_MAXIMO_ALVO = 64
    TAMANHO_LOTE = 16
    NUMERO_EPOCAS = 100
    TAXA_APRENDIZADO = 3e-4
    DECAIMENTO_PESO = 0.01
    
    # others/outros
    PREFIXO_TASK = "traduzir portugues para logica: "
    SEMENTE = 42
    DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
