import json
import torch
import os
import sys
import resource
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from configuracao import Configuracao
from pre_processamento import PreProcessador

class CarregadorDados:
    def __init__(self, configuracao):
        self.configuracao = configuracao
        self.tokenizador = AutoTokenizer.from_pretrained(configuracao.NOME_MODELO, use_fast=False)
        self.pre_processador = PreProcessador()
    
    def carregar(self, caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            arquivo = json.load(f)
            dados = arquivo["data"]
        
        lista_input = []
        lista_output = []
        
        for item in dados:
            lista_input.append(item[0])
            lista_output.append(item[1])
            
        dicionario = {
            "input": lista_input, 
            "output": lista_output
        }
        return Dataset.from_dict(dicionario)
    
    def funcao_preprocessamento(self, exemplos):
        entradas_originais = exemplos["input"]
        entradas_limpas = []
        for e in entradas_originais:
            limpo = self.pre_processador.limpar_texto(e)
            entradas_limpas.append(limpo)
        
        alvos_originais = exemplos["output"]
        
        entradas_preparadas = []
        for e in entradas_limpas:
            final = self.configuracao.PREFIXO_TASK + e
            entradas_preparadas.append(final)
        
        entradas_modelo = self.tokenizador(
            entradas_preparadas, 
            max_length=self.configuracao.COMPRIMENTO_MAXIMO_ENTRADA, 
            truncation=True, 
            padding="max_length"
        )
        
        rotulos_modelo = self.tokenizador(
            alvos_originais, 
            max_length=self.configuracao.COMPRIMENTO_MAXIMO_ALVO, 
            truncation=True, 
            padding="max_length"
        )
        
        lista_labels = []
        token_ids_alvo = rotulos_modelo["input_ids"]
        
        for label in token_ids_alvo:
            nova_label = []
            for l in label:
                if l != self.tokenizador.pad_token_id:
                    nova_label.append(l)
                else:
                    nova_label.append(-100)
            lista_labels.append(nova_label)
        
        entradas_modelo["labels"] = lista_labels
        return entradas_modelo
    
    def obter_datasets(self):
        ds_treino_bruto = self.carregar(self.configuracao.CAMINHO_CONJUNTO_TRAIN)
        ds_treino = ds_treino_bruto.map(self.funcao_preprocessamento, batched=True)
        
        ds_teste_bruto = self.carregar(self.configuracao.CAMINHO_CONJUNTO_TEST)
        ds_teste = ds_teste_bruto.map(self.funcao_preprocessamento, batched=True)
        
        return ds_treino, ds_teste

class ModuloTreinamento:
    def __init__(self, configuracao, modelo, ds_treino, ds_teste, tokenizador):
        self.configuracao = configuracao
        self.modelo = modelo
        self.ds_treino = ds_treino
        self.ds_teste = ds_teste
        self.tokenizador = tokenizador

    def calcular_metricas(self, pred):
        from sklearn.metrics import accuracy_score, f1_score
        
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        decoded_preds_bruto = self.tokenizador.batch_decode(pred_ids, skip_special_tokens=True)
        
        for i in range(len(labels_ids)):
            for j in range(len(labels_ids[i])):
                if labels_ids[i][j] == -100:
                    labels_ids[i][j] = self.tokenizador.pad_token_id
                    
        decoded_labels_bruto = self.tokenizador.batch_decode(labels_ids, skip_special_tokens=True)
        
        decoded_preds = []
        for p in decoded_preds_bruto:
            decoded_preds.append(p.strip())
            
        decoded_labels = []
        for l in decoded_labels_bruto:
            decoded_labels.append(l.strip())
        
        acc = accuracy_score(decoded_labels, decoded_preds)
        f1 = f1_score(decoded_labels, decoded_preds, average='macro', zero_division=0)
        
        return {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4)
        }

    def treinar(self):
        args = Seq2SeqTrainingArguments(
            output_dir=self.configuracao.DIRETORIO_SAIDA,
            evaluation_strategy="epoch",
            learning_rate=self.configuracao.TAXA_APRENDIZADO,
            per_device_train_batch_size=self.configuracao.TAMANHO_LOTE,
            per_device_eval_batch_size=self.configuracao.TAMANHO_LOTE,
            gradient_accumulation_steps=self.configuracao.GRADIENT_ACCUMULATION,
            gradient_checkpointing=True,
            num_train_epochs=self.configuracao.NUMERO_EPOCAS,
            weight_decay=self.configuracao.DECAIMENTO_PESO,
            save_total_limit=3,
            logging_steps=10,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
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
        
        caminho_hist = os.path.join(self.configuracao.DIRETORIO_SAIDA, "historico_treinamento.json")
        with open(caminho_hist, "w") as f:
            json.dump(treinador.state.log_history, f)
            
        return treinador

def main():
    # Limitar RAM para 15GB (15 * 1024 * 1024 * 1024 bytes)
    try:
        limite_ram = 15 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limite_ram, limite_ram))
        print(f"Limite de RAM definido para 15GB.")
    except Exception as e:
        print(f"Não foi possível definir o limite de RAM: {e}")

    configuracao = Configuracao()
    torch.manual_seed(configuracao.SEMENTE)
    
    carregador = CarregadorDados(configuracao)
    datasets = carregador.obter_datasets()
    ds_treino = datasets[0]
    ds_teste = datasets[1]
    
    modelo_obj = AutoModelForSeq2SeqLM.from_pretrained(
        configuracao.NOME_MODELO, 
        low_cpu_mem_usage=True
    ).to(configuracao.DISPOSITIVO)
    
    modulo = ModuloTreinamento(
        configuracao, 
        modelo_obj, 
        ds_treino, 
        ds_teste, 
        carregador.tokenizador
    )
    treinador = modulo.treinar()
    
    treinador.save_model(configuracao.DIRETORIO_SAIDA)
    carregador.tokenizador.save_pretrained(configuracao.DIRETORIO_SAIDA)

if __name__ == "__main__":
    main()
