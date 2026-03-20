import json
import torch
import os
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from configuracao import Configuracao

class CarregadorDados:
    """
    Carrega os arquivos JSON de treino e teste e realiza o pré-processamento.
    Loads the training and test JSON files and performs preprocessing.
    """
    def __init__(self, configuracao):
        self.configuracao = configuracao
        self.tokenizador = AutoTokenizer.from_pretrained(configuracao.NOME_MODELO, use_fast=False)
    
    def carregar(self, caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)["data"]
        dicionario = {"input": [item[0] for item in dados], "output": [item[1] for item in dados]}
        return Dataset.from_dict(dicionario)
    
    def funcao_preprocessamento(self, exemplos):
        entradas = [self.configuracao.PREFIXO_TASK + e for e in exemplos["input"]]
        alvos = exemplos["output"]
        
        entradas_modelo = self.tokenizador(entradas, max_length=self.configuracao.COMPRIMENTO_MAXIMO_ENTRADA, truncation=True, padding="max_length")
        rotulos = self.tokenizador(alvos, max_length=self.configuracao.COMPRIMENTO_MAXIMO_ALVO, truncation=True, padding="max_length")
        
        labels = rotulos["input_ids"]
        labels = [[(l if l != self.tokenizador.pad_token_id else -100) for l in label] for label in labels]
        
        entradas_modelo["labels"] = labels
        return entradas_modelo
    
    def obter_datasets(self):
        ds_treino = self.carregar(self.configuracao.CAMINHO_CONJUNTO_TRAIN).map(self.funcao_preprocessamento, batched=True)
        ds_teste = self.carregar(self.configuracao.CAMINHO_CONJUNTO_TEST).map(self.funcao_preprocessamento, batched=True)
        return ds_treino, ds_teste

class ModuloTreinamento:
    """
    Gerencia o processo de treinamento e avaliação com suporte a Seq2Seq.
    Manages the training and evaluation process with Seq2Seq support.
    """
    def __init__(self, configuracao, modelo, ds_treino, ds_teste, tokenizador):
        self.configuracao = configuracao
        self.modelo = modelo
        self.ds_treino = ds_treino
        self.ds_teste = ds_teste
        self.tokenizador = tokenizador
        self.metric = evaluate.load("rouge")

    def calcular_metricas(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        decoded_preds = self.tokenizador.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizador.pad_token_id
        decoded_labels = self.tokenizador.batch_decode(labels_ids, skip_special_tokens=True)
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    def treinar(self):
        args = Seq2SeqTrainingArguments(
            output_dir=self.configuracao.DIRETORIO_SAIDA,
            evaluation_strategy="epoch",
            learning_rate=self.configuracao.TAXA_APRENDIZADO,
            per_device_train_batch_size=self.configuracao.TAMANHO_LOTE,
            per_device_eval_batch_size=self.configuracao.TAMANHO_LOTE,
            num_train_epochs=self.configuracao.NUMERO_EPOCAS,
            weight_decay=self.configuracao.DECAIMENTO_PESO,
            save_total_limit=3,
            logging_steps=10,
            predict_with_generate=True,
            fp16=False, # AMD ROCm might work better without fp16 unless specifically configured
            report_to="none"
        )
        
        treinador = Seq2SeqTrainer(
            model=self.modelo,
            args=args,
            train_dataset=self.ds_treino,
            eval_dataset=self.ds_teste,
            processing_class=self.tokenizador,
            compute_metrics=self.calcular_metricas
        )
        
        treinador.train()
        
        with open(os.path.join(self.configuracao.DIRETORIO_SAIDA, "historico_treinamento.json"), "w") as f:
            json.dump(treinador.state.log_history, f)
            
        return treinador

def main():
    configuracao = Configuracao()
    torch.manual_seed(configuracao.SEMENTE)
    
    carregador = CarregadorDados(configuracao)
    ds_treino, ds_teste = carregador.obter_datasets()
    
    modelo_obj = AutoModelForSeq2SeqLM.from_pretrained(configuracao.NOME_MODELO)
    
    modulo = ModuloTreinamento(configuracao, modelo_obj, ds_treino, ds_teste, carregador.tokenizador)
    treinador = modulo.treinar()
    
    treinador.save_model(configuracao.DIRETORIO_SAIDA)
    carregador.tokenizador.save_pretrained(configuracao.DIRETORIO_SAIDA)

if __name__ == "__main__":
    main()
