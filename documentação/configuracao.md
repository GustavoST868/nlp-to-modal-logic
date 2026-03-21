# Documentação: `configuracao.py`

O arquivo `configuracao.py` desempenha o papel fundamental de "cérebro" das definições deste projeto. Ele centraliza todas as constantes e hiperparâmetros necessários para que tanto o treinamento quanto o teste do modelo funcionem em harmonia. Ao utilizar uma classe única chamada `Configuracao`, o código garante que alterações feitas em um ponto do sistema (como o nome do modelo base ou o caminho do dataset) sejam refletidas automaticamente em todos os outros scripts que importam esse arquivo.

Esta arquitetura facilita muito a manutenção, pois evita que valores "mágicos" ou repetidos fiquem espalhados pelo código. Através deste arquivo, o desenvolvedor pode ajustar desde limites técnicos de hardware até as preferências específicas de como o modelo mT5 deve interpretar os dados de entrada (como o prefixo de tradução).

A tabela abaixo detalha as principais variáveis de configuração e seus respectivos propósitos:

| Variável | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| `CAMINHO_CONJUNTO_TRAIN` | `"dataset.json"` | Caminho do arquivo JSON com os dados para treinamento do modelo. |
| `CAMINHO_CONJUNTO_TEST` | `"dataset_teste.json"` | Caminho do arquivo JSON com os dados para validação e métricas. |
| `NOME_MODELO` | `"google/mt5-base"` | Identificador do modelo pré-treinado na Hugging Face que será usado como base. |
| `DIRETORIO_SAIDA` | `"./modelo_final"` | Pasta onde os pesos treinados e o tokenizador serão salvos. |
| `TAMANHO_LOTE` | `4` | Quantidade de exemplos processados simultaneamente (Batch Size). |
| `NUMERO_EPOCAS` | `15` | Total de repetições completas sobre o dataset durante o treino. |
| `TAXA_APRENDIZADO` | `1e-4` | Velocidade com que o modelo ajusta seus parâmetros diante dos erros. |
| `GRADIENT_ACCUMULATION`| `4` | Número de passos antes de atualizar os pesos, estabilizando lotes pequenos. |
| `PREFIXO_TASK` | `"traduzir portugues..."`| Texto de comando que ensina o modelo qual tarefa realizar. |
| `DISPOSITIVO` | `cuda` ou `cpu` | Detecta automaticamente se o processamento será via placa de vídeo ou processador. |

Para personalizar o projeto, basta alterar os valores desta classe. Por exemplo, se o seu computador possuir menos de 16GB de memória RAM, é recomendável reduzir o `TAMANHO_LOTE` para valores menores (como 1 ou 2). Se desejar maior precisão em sentenças complexas, o aumento do `NUMERO_EPOCAS` pode ajudar na convergência do modelo.
