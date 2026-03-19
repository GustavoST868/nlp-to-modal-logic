import json
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from configuracao import Configuracao

class CarregadorDados:
    def __init__(self, configuracao):
        self.configuracao = configuracao
        self.tokenizador = AutoTokenizer.from_pretrained(configuracao.NOME_MODELO)
    
    def carregar_conjunto_dados(self):
        with open(self.configuracao.CAMINHO_CONJUNTO_DADOS, 'r', encoding='utf-8') as f:
            dados = json.load(f)["data"]
        
        # convert to dict for dataset/converter para dict para dataset
        dicionario_conjunto = {"input": [item[0] for item in dados], "output": [item[1] for item in dados]}
        conjunto_dados = Dataset.from_dict(dicionario_conjunto)
        return conjunto_dados
    
    def funcao_preprocessamento(self, exemplos):
        entradas = [self.configuracao.PREFIXO_TASK + e for e in exemplos["input"]]
        alvos = exemplos["output"]
        
        entradas_modelo = self.tokenizador(entradas, max_length=self.configuracao.COMPRIMENTO_MAXIMO_ENTRADA, truncation=True, padding="max_length")
        rotulos = self.tokenizador(alvos, max_length=self.configuracao.COMPRIMENTO_MAXIMO_ALVO, truncation=True, padding="max_length")
        
        # mask padding token in labels so that the trainer ignores them/mascarar o token de preenchimento (padding) nos rótulos para que o trainer os ignore
        labels = rotulos["input_ids"]
        labels = [[(l if l != self.tokenizador.pad_token_id else -100) for l in label] for label in labels]
        
        entradas_modelo["labels"] = labels
        return entradas_modelo
    
    def obter_conjunto_processado(self):
        conjunto_dados = self.carregar_conjunto_dados()
        conjunto_processado = conjunto_dados.map(self.funcao_preprocessamento, batched=True)
        return conjunto_processado

class Modelo:
    def __init__(self, configuracao):
        self.configuracao = configuracao
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(configuracao.NOME_MODELO)
    
    def obter_modelo(self):
        return self.modelo

class ModuloTreinamento:
    def __init__(self, configuracao, modelo, conjunto_dados, tokenizador):
        self.configuracao = configuracao
        self.modelo = modelo
        self.conjunto_dados = conjunto_dados
        self.tokenizador = tokenizador
    
    def treinar(self):
        argumentos_treinamento = TrainingArguments(
            output_dir=self.configuracao.DIRETORIO_SAIDA,
            eval_strategy="epoch",
            learning_rate=self.configuracao.TAXA_APRENDIZADO,
            per_device_train_batch_size=self.configuracao.TAMANHO_LOTE,
            per_device_eval_batch_size=self.configuracao.TAMANHO_LOTE,
            num_train_epochs=self.configuracao.NUMERO_EPOCAS,
            weight_decay=self.configuracao.DECAIMENTO_PESO,
            save_total_limit=3,
            save_steps=50,
            logging_steps=10,
        )
        
        treinador = Trainer(
            model=self.modelo,
            args=argumentos_treinamento,
            train_dataset=self.conjunto_dados,
            eval_dataset=self.conjunto_dados, 
            processing_class=self.tokenizador,
        )
        
        treinador.train()
        return treinador

def main():
    configuracao = Configuracao()
    torch.manual_seed(configuracao.SEMENTE)
    
    print("Iniciando carregamento de dados...")
    carregador_dados = CarregadorDados(configuracao)
    conjunto_dados = carregador_dados.obter_conjunto_processado()
    
    print("Iniciando carregamento do modelo...")
    modulo_modelo = Modelo(configuracao)
    modelo = modulo_modelo.obter_modelo()
    
    print("Iniciando treinamento...")
    modulo_treinamento = ModuloTreinamento(configuracao, modelo, conjunto_dados, carregador_dados.tokenizador)
    treinador = modulo_treinamento.treinar()
    
    # save the final model in the configured directory/salvar o modelo final no diretório configurado
    treinador.save_model(configuracao.DIRETORIO_SAIDA)
    # also save the tokenizer separately to ensure/também salvar o tokenizador separadamente para garantir
    carregador_dados.tokenizador.save_pretrained(configuracao.DIRETORIO_SAIDA)
    
    print("-" * 30)
    print(f"Treinamento concluído e modelo final salvo em: {configuracao.DIRETORIO_SAIDA}")
    print("-" * 30)

if __name__ == "__main__":
    main()
