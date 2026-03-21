#!/bin/bash

VENV_DIR="venv"

verificar_hardware_e_instalar() {
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
