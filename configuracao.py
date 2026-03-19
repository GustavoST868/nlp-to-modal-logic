import torch

class Configuracao:
    # Caminhos
    CAMINHO_CONJUNTO_DADOS = "dataset.json"
    NOME_MODELO = "google/mt5-small"  # Modelo base para seq2seq multilíngue
    DIRETORIO_SAIDA = "./modelo_final"
    
    # Hiperparâmetros de treinamento
    COMPRIMENTO_MAXIMO_ENTRADA = 64
    COMPRIMENTO_MAXIMO_ALVO = 64
    TAMANHO_LOTE = 16
    NUMERO_EPOCAS = 100
    TAXA_APRENDIZADO = 3e-4
    DECAIMENTO_PESO = 0.01
    
    # Outros
    PREFIXO_TASK = "traduzir portugues para logica: "
    SEMENTE = 42
    DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
