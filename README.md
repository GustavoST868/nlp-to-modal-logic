# Treinamento de LLM para Lógica Modal

Este projeto implementa um sistema modular para treinar um modelo de linguagem (LLM) de maneira supervisionada, convertendo descrições em linguagem natural (em português) para fórmulas de lógica modal e predicativa. O objetivo é demonstrar como LLMs podem ser fine-tuned para tarefas de tradução simbólica, aplicando conceitos de lógica modal a cenários cotidianos.

## Teoria por Trás do Projeto

### Lógica Modal
A lógica modal estende a lógica clássica com operadores modais como:
- **□ (necessidade)**: Indica que algo é necessariamente verdadeiro (ex.: "É necessário respirar para viver").
- **◇ (possibilidade)**: Indica que algo pode ser verdadeiro (ex.: "É possível que chova amanhã").

Esses operadores permitem modelar conceitos como obrigatoriedade, possibilidade e conhecimento em contextos normais, como regras diárias, probabilidades e obrigações.

### Modelo de Linguagem (T5)
O projeto usa o modelo T5 (Text-to-Text Transfer Transformer) da Hugging Face, que trata todas as tarefas de NLP como geração de texto. Aqui, é fine-tuned para seq2seq (sequência-para-sequência), mapeando frases em português para expressões lógicas.

- **Fine-tuning supervisionado**: O modelo aprende a traduzir entradas naturais para saídas simbólicas usando pares de treinamento.
- **Arquitetura**: Encoder-decoder, ideal para tarefas de tradução.
- **Base**: T5-small (aprox. 60 milhões de parâmetros), adequado para protótipos e datasets pequenos.

### Treinamento e Inferência
- **Treinamento**: Usa a biblioteca Transformers para otimizar o modelo com dados supervisionados. Hiperparâmetros são configuráveis em `config.py`.
- **Inferência**: O modelo gera fórmulas lógicas a partir de texto natural, usando beam search para melhorar a qualidade.
- **Continuando o Treinamento**: Adicione novos exemplos ao `dataset.json` e execute `python main.py` novamente. Para datasets maiores, ajuste epochs e batch size.

## Estrutura do Repositório

```
Algoritmos LLM/
├── app.py                          # Interface web com Streamlit para inferência interativa
├── config.py                       # Configurações do treinamento (hiperparâmetros, caminhos)
├── data_loader.py                  # Carrega e processa o dataset.json para treinamento
├── dataset.json                    # Conjunto de dados: pares [frase natural, fórmula lógica]
├── interface_tkinter.py            # Interface gráfica desktop com Tkinter
├── main.py                         # Script principal para iniciar o treinamento
├── model.py                        # Define o modelo T5 e funções de geração
├── README.md                       # Este arquivo (documentação)
├── requirements.txt                # Dependências Python
├── salvar_modelo.py                # Salva o modelo treinado em disco
├── test_model.py                   # Testa o modelo com exemplos pré-definidos
├── trainer.py                      # Classe para gerenciar o treinamento com Transformers
├── __pycache__/                    # Cache Python (ignorado)
├── modelo_treinado/                # Modelo treinado salvo (checkpoints)
│   ├── checkpoint-130/             # Checkpoint intermediário
│   ├── ...
│   └── checkpoint-150/
├── results/                        # Modelo final treinado (usado para inferência)
│   ├── checkpoint-140/             # Checkpoints finais
│   ├── ...
│   └── checkpoint-160/
└── venv/                           # Ambiente virtual (não versionado)
```

### Funcionalidade dos Arquivos

- **`app.py`**: Cria uma interface web simples com Streamlit. Permite inserir frases e ver a tradução lógica em tempo real. Execute com `streamlit run app.py`.
- **`config.py`**: Contém constantes como taxa de aprendizado, número de epochs, tamanho do lote e caminhos de diretórios. Edite para ajustar o treinamento.
- **`data_loader.py`**: Carrega o `dataset.json`, tokeniza os dados e prepara datasets para o treinamento (usando Hugging Face Datasets).
- **`dataset.json`**: Dataset principal. Contém pares de treinamento. Recentemente expandido com exemplos básicos de lógica modal aplicados a situações normais (ex.: possibilidade de chuva, necessidade de estudo).
- **`interface_tkinter.py`**: Interface desktop com Tkinter. Similar ao app.py, mas em GUI nativa. Execute com `python interface_tkinter.py`.
- **`main.py`**: Ponto de entrada para o treinamento. Carrega configurações, dados e modelo, então inicia o treino via `trainer.py`.
- **`model.py`**: Define funções para carregar o modelo T5, tokenizar e gerar saídas. Usado em inferência.
- **`requirements.txt`**: Lista de pacotes necessários (ex.: transformers, torch, streamlit). Instale com `pip install -r requirements.txt`.
- **`salvar_modelo.py`**: Após o treinamento, salva o modelo em `modelo_treinado/` e opcionalmente zipa para distribuição.
- **`test_model.py`**: Executa testes com frases de exemplo, imprimindo entradas e saídas geradas. Útil para validação rápida.
- **`trainer.py`**: Classe `ModuloTreinamento` que configura e executa o treinamento usando `Trainer` do Transformers.
- **`modelo_treinado/` e `results/`**: Diretórios com modelos salvos. `results/` é o modelo final para uso em produção.

## Como Usar

### Configuração Inicial
1. **Clone ou baixe o repositório**.
2. **Crie um ambiente virtual**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. **Instale dependências**:
   ```bash
   pip install -r requirements.txt
   ```

### Treinamento do Modelo
1. **Prepare o dataset**: Edite `dataset.json` para adicionar/remover exemplos. O formato é JSON com lista de pares [entrada, saída].
2. **Execute o treinamento**:
   ```bash
   python main.py
   ```
   - Isso treina o modelo com os dados atuais. Checkpoints são salvos em `results/`.
3. **Salve o modelo** (opcional):
   ```bash
   python salvar_modelo.py
   ```
   - Salva em `modelo_treinado/` e cria um ZIP.

### Continuando o Treinamento
- Para retreinar com novos dados: Atualize `dataset.json`, então execute `python main.py` novamente. O modelo será carregado do checkpoint mais recente em `results/` e continuará o treino.
- Ajuste hiperparâmetros em `config.py` para datasets maiores (ex.: aumente `NUMERO_EPOCAS`).

### Teste e Inferência
1. **Teste via linha de comando**:
   ```bash
   python test_model.py
   ```
   - Testa com exemplos hardcoded.
2. **Interface desktop**:
   ```bash
   python interface_tkinter.py
   ```

### Exemplo de Uso
- Entrada: "É possível que chova amanhã"
- Saída esperada: "◇(chover_amanhã)"

## Notas Adicionais
- **Dataset**: Inclui relações familiares, profissões e agora exemplos modais básicos. Expanda para melhorar o desempenho.
- **Limitações**: Modelo pequeno; para produção, use T5-base ou larger. Treinamento requer GPU para eficiência.
- **Contribuições**: Adicione exemplos ao dataset ou melhore interfaces. Teste sempre após mudanças.
- **Dependências**: Certifique-se de ter PyTorch compatível com CUDA se usar GPU.

Para dúvidas, consulte a documentação do Hugging Face Transformers.