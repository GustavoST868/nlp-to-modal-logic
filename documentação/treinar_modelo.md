# Documentação: `treinar_modelo.py`

O script `treinar_modelo.py` é o núcleo tecnológico deste projeto, sendo responsável por todo o processo de fine-tuning do modelo base **mT5-base** da Google. Este processo transforma um modelo de linguagem genérico em um especialista em traduzir sentenças em português para estruturas de lógica modal (necessidade e possibilidade). O script é dividido em classes bem definidas que cuidam desde a limpeza e preparação dos dados até o ciclo completo de treinamento e salvamento dos pesos finais.

Para possibilitar o treinamento em máquinas com até 15GB de RAM, o `treinar_modelo.py` implementa técnicas modernas e eficientes, como o `Gradient Checkpointing` e o acúmulo de gradientes (`Gradient Accumulation`). O script também monitora continuamente o desempenho do modelo, calculando métricas de acurácia (Exact Match) e pontuação F1 a cada época do treinamento. Isso garante que o modelo não apenas aprenda o formato da saída, mas que sua tradução sintática seja idêntica ao esperado em termos de lógica modal rigorosa.

Abaixo, apresentamos uma tabela com os principais componentes e responsabilidades do script:

| Componente | Tipo | Responsabilidade Principal |
| :--- | :--- | :--- |
| `CarregadorDados`| Classe | Carrega os arquivos JSON (`dataset.json`) e prepara os tokens de entrada. |
| `ModuloTreinamento`| Classe | Configura o motor `Seq2SeqTrainer` da Hugging Face com otimizações. |
| `calcular_metricas`| Método | Calcula o **Exact Match** (Acurácia) e o **F1 Score** entre o modelo e os rótulos. |
| `Gradient Checkpointing`| Técnica | Reduz drasticamente o consumo de VRAM ao não armazenar ativações intermediárias. |
| `FP16` | Técnica | Utiliza precisão mista de 16-bits para dobrar a velocidade do treino em GPUs compatíveis. |
| `main()` | Função | Orquestra o ciclo completo: carrega dados, treina, avalia e salva resultados. |
| `./modelo_final` | Diretório| Salva o modelo treinado, o tokenizador e o histórico de logs. |

Para iniciar o treinamento, certifique-se de ter o ambiente configurado, ative o ambiente virtual (`source venv/bin/activate`) e execute `python3 treinar_modelo.py`. O tempo de processamento varia conforme o hardware, mas o script foi desenhado para ser resiliente e eficiente mesmo em computadores com recursos de memória limitados.
