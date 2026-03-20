#!/bin/bash

# Este script configura o ambiente de desenvolvimento, detectando se o hardware é AMD ou outro (NVIDIA/CPU) para instalar as dependências corretas do PyTorch.
# This script sets up the development environment, detecting if the hardware is AMD or another (NVIDIA/CPU) to install the correct PyTorch dependencies.

VENV_DIR="venv"

verificar_hardware_e_instalar() {
    # Detecta o hardware e instala as dependências necessárias do PyTorch.
    # Detects hardware and installs necessary PyTorch dependencies.
    
    echo "Verificando hardware..."
    if lspci | grep -i "vga" | grep -i "AMD" > /dev/null; then
        echo "Hardware AMD detectado. Instalando PyTorch com suporte a ROCm..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
    else
        echo "Hardware NVIDIA ou apenas CPU detectado. Instalando PyTorch padrão..."
        pip install torch torchvision torchaudio
    fi
}

configurar_ambiente() {
    # Cria o ambiente virtual, ativa e instala as dependências do projeto.
    # Creates the virtual environment, activates it, and installs project dependencies.
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    verificar_hardware_e_instalar

    if [ -f "requirements.txt" ]; then
        grep -v "torch" requirements.txt > temp_requirements.txt
        pip install -r temp_requirements.txt
        rm temp_requirements.txt
    fi
}

configurar_ambiente
