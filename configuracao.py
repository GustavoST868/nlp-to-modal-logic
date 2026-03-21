import torch

class Configuracao:
    """
    Define todas as variáveis de configuração, hiperparâmetros e caminhos para o projeto de lógica modal.
    Defines all configuration variables, hyperparameters and paths for the modal logic project.
    """
    CAMINHO_CONJUNTO_TRAIN = "dataset.json"
    CAMINHO_CONJUNTO_TEST = "dataset_teste.json"
    NOME_MODELO = "google/mt5-base"
    DIRETORIO_SAIDA = "./modelo_final"
    
    COMPRIMENTO_MAXIMO_ENTRADA = 128
    COMPRIMENTO_MAXIMO_ALVO = 128

    TAMANHO_LOTE = 4            
    NUMERO_EPOCAS = 15           

    TAXA_APRENDIZADO = 1e-4      
    DECAIMENTO_PESO = 0.01

    WARMUP_STEPS = 500          
    GRADIENT_ACCUMULATION = 4  

    PREFIXO_TASK = "traduzir portugues para logica: "

    SEMENTE = 42
    DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
