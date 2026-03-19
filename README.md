# Treinamento de LLM para Lógica Modal

Este projeto implementa um sistema modular para treinar um modelo de linguagem (LLM) de maneira supervisionada, convertendo descrições em linguagem natural (em português) para fórmulas de lógica modal e predicativa. O objetivo é demonstrar como LLMs podem ser fine-tuned para tarefas de tradução simbólica, aplicando conceitos de lógica modal a cenários cotidianos.

## Teoria por Trás do Projeto

### Lógica Modal
A lógica modal estende a lógica clássica com operadores modais como:
- **□ (necessidade)**: Indica que algo é necessariamente verdadeiro (ex.: "É necessário respirar para viver").
- **◇ (possibilidade)**: Indica que algo pode ser verdadeiro (ex.: "É possível que chova amanhã").

Esses operadores permitem modelar conceitos como obrigatoriedade, possibilidade e conhecimento em contextos normais, como regras diárias, probabilidades e obrigações.

### Modelo de Linguagem (mt5-small)
O projeto utiliza o modelo **mt5-small** (Multilingual Text-to-Text Transfer Transformer) da Hugging Face. O T5 trata todas as tarefas de NLP como geração de texto. Neste projeto, ele é fine-tuned para seq2seq (sequência-para-sequência), mapeando frases em português para expressões lógicas.

- **Fine-tuning supervisionado**: O modelo aprende a traduzir entradas naturais para saídas simbólicas usando pares de treinamento.
- **Arquitetura**: Encoder-decoder, ideal para tarefas de tradução.
- **Base**: `google/mt5-small`, adequado para protótipos e datasets pequenos devido ao seu tamanho reduzido e eficiência.

## Estrutura do Repositório

```text
Algoritmos LLM/
├── modelo_final/
├── modelo_treinado/
├── results/
├── venv/
├── configuracao.py
├── configurar_ambiente.sh
├── dataset.json
├── README.md
├── requirements.txt
├── testar_modelo.py
└── treinar_modelo.py
```

### Funcionalidade das Pastas e Arquivos

- **`modelo_final/`**: Diretório que armazena o modelo T5 após o fim do treinamento, pronto para uso.
- **`modelo_treinado/`**: Contém os checkpoints salvos automaticamente durante o processo de treino.
- **`results/`**: Pasta gerada pelo Trainer para armazenar logs de métricas e progresso.
- **`venv/`**: Ambiente virtual que isola as dependências do projeto do restante do sistema.
- **`configuracao.py`**: Centraliza todas as variáveis técnicas (épocas, batch size, learning rate) e caminhos globais.
- **`configurar_ambiente.sh`**: Automatiza a criação da `venv` e a instalação dos pacotes necessários via `pip`.
- **`dataset.json`**: Arquivo base com exemplos estruturados de linguagem natural e suas respectivas fórmulas lógicas.
- **`README.md`**: Documentação técnica contendo instruções de configuração, teoria e uso do projeto.
- **`requirements.txt`**: Lista detalhada de bibliotecas externas (Transformers, PyTorch, etc.) exigidas.
- **`testar_modelo.py`**: Interface interativa em Tkinter para realizar inferências rápidas com o modelo treinado.
- **`treinar_modelo.py`**: Motor principal que carrega os dados, inicializa o modelo e executa o fine-tuning.

## Como Usar

### 1. Configuração Inicial
O projeto inclui um script de automação para facilitar o setup inicial em ambiente Linux:

```bash
chmod +x configurar_ambiente.sh
./configurar_ambiente.sh
```

Isso criará a pasta `venv` e instalará tudo o que é necessário. Caso queira fazer manualmente:
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

### 2. Treinamento do Modelo
Para treinar o modelo com os dados do `dataset.json`:

```bash
python treinar_modelo.py
```
- O modelo será salvo automaticamente em `modelo_final/` após o término das épocas configuradas.
- Checkpoints intermediários ficarão em `modelo_treinado/`.

### 3. Teste e Inferência
Para utilizar o modelo treinado através da interface visual:

```bash
python testar_modelo.py
```
1. A interface carregará o modelo de `modelo_final/`.
2. Digite uma frase (ex: "Talvez chova amanhã") e clique em **GERAR TRADUÇÃO**.
3. O resultado aparecerá na caixa de texto inferior.

## Personalização
- **Adicionar Dados**: Edite o `dataset.json` seguindo o formato existente e execute `treinar_modelo.py` novamente para que o modelo aprenda os novos padrões.
- **Ajustar Qualidade**: Se as traduções não estiverem precisas, aumente o `NUMERO_EPOCAS` ou adicione mais polimento ao `dataset.json` em `configuracao.py`.

## Notas Técnicas
- **Dispositivo**: O código detecta automaticamente se você possui uma GPU (CUDA) disponível para acelerar o treino; caso contrário, usará a CPU.
- **mt5-small**: Escolhido por ser multilíngue e leve, ideal para rodar em hardware doméstico sem necessidade de recursos massivos.