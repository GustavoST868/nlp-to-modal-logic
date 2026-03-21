# Documentação: `testar_modelo.py`

O script `testar_modelo.py` fornece uma interface amigável e interativa em Flask para utilizar o modelo de tradução de lógica modal já treinado. Este componente web permite que o usuário teste sentenças em português e veja a tradução para fórmulas lógicas (utilizando os operadores de necessidade □ e possibilidade ◇) em tempo real. Ele também permite que o desenvolvedor acompanhe o desempenho histórico do treinamento do modelo.

Ao carregar o modelo de forma inteligente (Lazy Loading), o script economiza recursos de hardware durante a inicialização. Para tornar o uso mais fluido, o `testar_modelo.py` também inclui um temporizador que abre automaticamente o navegador padrão do sistema na interface web (`http://127.0.0.1:5000/`). Isso remove a necessidade de o usuário procurar o link no terminal. Todas as traduções são feitas via uma API interna que comunica o frontend HTML com os pesos do modelo mT5.

Confira abaixo a tabela com os principais endpoints e funcionalidades do servidor Flask:

| Endpoint | Método | Descrição | Parâmetros (JSON/Response) |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Serve a página inicial da interface web. | Retorna o arquivo `index.html`. |
| `/api/translate` | `POST` | Traduz o texto de entrada para lógica modal. | Entrada: `{"text": "sentença"}` / Saída: `{"output": "□..."}` |
| `/api/performance`| `GET` | Recupera o histórico detalhado do treino. | Retorna o arquivo JSON com perda (loss) e acurácia. |
| `carregar_modelo` | `Interno`| Carrega o tokenizador e os pesos do modelo. | Busca os arquivos no diretório `./modelo_final`. |
| `open_browser` | `Externo`| Abre o navegador automaticamente. | Utiliza a biblioteca `webbrowser`. |

Para testar o modelo, basta ativar o ambiente virtual (`source venv/bin/activate`) e executar `python3 testar_modelo.py`. O sistema foi desenvolvido para ser extremamente fácil de usar, fornecendo uma ponte entre o mundo acadêmico da lógica modal e uma aplicação prática e visual.
