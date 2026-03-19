#!/bin/bash

# Nome do diretório do ambiente virtual
VENV_DIR="venv"

echo "=========================================================="
echo "Configurando o ambiente para o projeto de Algoritmos LLM..."
echo "=========================================================="

# 1. Verificar se o Python 3 está instalado
if ! command -v python3 &> /dev/null
then
    echo "Erro: python3 não encontrado. Por favor, instale o Python 3."
    exit 1
fi

# 2. Criar o ambiente virtual se ele não existir
if [ ! -d "$VENV_DIR" ]; then
    echo "-> Criando o ambiente virtual em '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
else
    echo "-> O ambiente virtual '$VENV_DIR' já existe."
fi

# 3. Ativar o ambiente virtual
echo "-> Ativando o ambiente virtual..."
source "$VENV_DIR/bin/activate"

# 4. Atualizar o pip
echo "-> Atualizando o pip..."
pip install --upgrade pip

# 5. Instalar as dependências do requirements.txt
if [ -f "requirements.txt" ]; then
    echo "-> Instalando dependências do arquivo 'requirements.txt'..."
    pip install -r requirements.txt
else
    echo "-> Aviso: 'requirements.txt' não encontrado. Nenhuma dependência instalada."
fi

echo "=========================================================="
echo "Configuração concluída com sucesso!"
echo "Para ativar o ambiente manualmente no futuro, execute:"
echo "source $VENV_DIR/bin/activate"
echo "=========================================================="
