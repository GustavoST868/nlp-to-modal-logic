from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
import webbrowser
from threading import Timer
from configuracao import Configuracao

app = Flask(__name__)

# Carregamento global do modelo
# Global model loading
config = Configuracao()
modelo = None
tokenizador = None
device = config.DISPOSITIVO

def carregar_modelo():
    """
    Carrega o modelo de forma preguiçosa (ou na inicialização).
    Loads the model lazily (or on initialization).
    """
    global modelo, tokenizador
    caminho_modelo = config.DIRETORIO_SAIDA
    if not os.path.exists(caminho_modelo) or not os.listdir(caminho_modelo):
        return False, "Modelo não encontrado no diretório de saída. Por favor, treine o modelo primeiro."
    
    try:
        tokenizador = AutoTokenizer.from_pretrained(caminho_modelo, use_fast=False)
        modelo = AutoModelForSeq2SeqLM.from_pretrained(caminho_modelo)
        modelo.to(device)
        return True, "Modelo carregado com sucesso!"
    except Exception as e:
        return False, f"Erro ao carregar o modelo: {str(e)}"

# Inicializa o modelo
success, message = carregar_modelo()
print(message)

@app.route('/')
def index():
    """Serve a página principal da interface web."""
    return render_template('index.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    """Endpoint para tradução de linguagem natural para lógica modal."""
    if not modelo:
        return jsonify({"error": "Modelo não carregado ou não encontrado."}), 500
    
    data = request.json
    texto = data.get('text', '').strip()
    
    if not texto:
        return jsonify({"error": "Texto vazio."}), 400
    
    try:
        # Preparação da entrada
        inp = tokenizador(config.PREFIXO_TASK + texto, return_tensors="pt").to(device)
        
        # Geração da saída
        with torch.no_grad():
            out = modelo.generate(**inp, max_length=128, num_beams=4)
            traducao = tokenizador.decode(out[0], skip_special_tokens=True)
            
        return jsonify({"input": texto, "output": traducao})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def performance():
    """Retorna os dados de desempenho do histórico de treinamento."""
    caminho_hist = os.path.join(config.DIRETORIO_SAIDA, "historico_treinamento.json")
    if not os.path.exists(caminho_hist):
        return jsonify({"error": "Histórico de treinamento não encontrado."}), 404
        
    try:
        with open(caminho_hist, "r") as f:
            historico = json.load(f)
        return jsonify(historico)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser():
    """Abre o navegador automaticamente após o servidor iniciar."""
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("\n--- Iniciando Tradutor de Lógica Modal (Interface Web) ---")
    print("Acesse em: http://127.0.0.1:5000/")
    
    # Timer para abrir o navegador
    Timer(1, open_browser).start()
    
    # Inicia o servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=False)
