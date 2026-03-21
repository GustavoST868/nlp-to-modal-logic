# Tradução de Linguagem Natural para Lógica Modal com LLM

Este projeto implementa um sistema de processamento de linguagem natural (NLP) para converter sentenças em português para fórmulas de lógica modal (usando operadores de necessidade □ e possibilidade ◇). O sistema utiliza um modelo **mT5-base** fine-tuned para tradução simbólica.

## 🧠 Teoria e Modelo

### Lógica Modal
Diferente da lógica proposicional clássica, a lógica modal introduz operadores que qualificam a verdade de uma sentença:
- **□ (Necessidade)**: O que deve ser verdade (ex: "É obrigatório que X").
- **◇ (Possibilidade)**: O que pode ser verdade (ex: "É possível que X").

### Modelo mT5-base
O projeto utiliza o **google/mt5-base**, um modelo encoder-decoder multilíngue da Hugging Face. Ele foi escolhido por sua robustez e capacidade de lidar com a complexidade da gramática portuguesa ao mapeá-la para estruturas lógicas rigorosas.

## 🚀 Otimizações de Memória (Limite de 15GB)

Para permitir o treinamento em máquinas com hardware limitado (como GPUs com 12GB-16GB de VRAM ou 16GB de RAM total), o projeto implementa várias técnicas de eficiência:

1.  **Gradient Checkpointing**: Reduz drasticamente o uso de VRAM ao recalcular ativações intermediárias em vez de armazená-las.
2.  **Acúmulo de Gradientes**: Permite treinar com lotes pequenos (`batch_size=4`), mas mantendo a estabilidade de lotes maiores através de acumulação (`accumulation_steps=4`).
3.  **Precisão Mista (FP16)**: Utiliza tensores de 16-bits para acelerar o treino e reduzir o consumo de memória pela metade (quando disponível).
4.  **Low CPU Memory Usage**: Carregamento otimizado para evitar picos de consumo de RAM do sistema durante a inicialização.

## 📂 Estrutura do Projeto

```text
nlp-to-modal-logic/
├── dataset.json            # Dados para treinamento
├── dataset_teste.json      # Dados para avaliação (validação)
├── configuracao.py         # Hiperparâmetros e constantes
├── treinar_modelo.py       # Script principal de fine-tuning
├── testar_modelo.py        # Interface interativa para inferência
├── modelo_final/           # Modelo treinado e salvo
├── requirements.txt        # Dependências do Python
└── configurar_ambiente.sh  # Script de setup automatizado
```

## 🛠️ Como Usar

### 1. Preparação do Ambiente
Execute o script de automação (disponível para Linux):
```bash
chmod +x configurar_ambiente.sh
./configurar_ambiente.sh
```

### 2. Treinamento
Para iniciar o treinamento e avaliação:
```bash
python3 treinar_modelo.py
```
O treino utiliza o `dataset.json` para aprendizado e o `dataset_teste.json` para medir a performance usando a métrica **ROUGE**. O modelo final será salvo em `modelo_final/`.

### 3. Inferência (Teste Manual)
Após o treino, você pode testar o modelo com a interface interativa:
```bash
python3 testar_modelo.py
```

## 📊 Avaliação e Métricas
O projeto utiliza a biblioteca `evaluate` da Hugging Face para calcular métricas ROUGE durante o treino, garantindo que a tradução sintática das fórmulas lógicas esteja o mais próxima possível do alvo esperado.

## ⚙️ Personalização
Você pode ajustar o comportamento do sistema editando `configuracao.py`:
- **TAMANHO_LOTE**: Aumente se tiver mais de 16GB de RAM.
- **NUMERO_EPOCAS**: Aumente para treinos mais longos e precisos.
- **NOME_MODELO**: Pode ser trocado por `mt5-small` para máquinas muito básicas ou `mt5-large` para máxima performance.