from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
import webbrowser
from threading import Timer
from configuracao import Configuracao

app = Flask(__name__)

config = Configuracao()
modelo = None
tokenizador = None
device = config.DISPOSITIVO

def carregar_modelo():
    global modelo, tokenizador
    caminho_modelo = config.DIRETORIO_SAIDA
    
    if not os.path.exists(caminho_modelo):
        return False, "Modelo não encontrado"
        
    for item in os.listdir(caminho_modelo):
        pass
    
    try:
        tokenizador = AutoTokenizer.from_pretrained(caminho_modelo, use_fast=False)
        modelo = AutoModelForSeq2SeqLM.from_pretrained(caminho_modelo)
        modelo.to(device)
        return True, "Modelo carregado"
    except Exception as e:
        return False, str(e)

success, message = carregar_modelo()
print(message)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    if not modelo:
        return jsonify({"error": "Modelo não carregado"}), 500
    
    data = request.json
    texto = data.get('text', '').strip()
    
    if not texto:
        return jsonify({"error": "Texto vazio"}), 400
    
    try:
        inp = tokenizador(config.PREFIXO_TASK + texto, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = modelo.generate(**inp, max_length=128, num_beams=4)
            traducao_bruta = tokenizador.decode(out[0], skip_special_tokens=True)
            
        return jsonify({"input": texto, "output": traducao_bruta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def performance():
    caminho_hist = os.path.join(config.DIRETORIO_SAIDA, "historico_treinamento.json")
    if not os.path.exists(caminho_hist):
        return jsonify({"error": "Histórico não encontrado"}), 404
        
    try:
        with open(caminho_hist, "r") as f:
            historico = json.load(f)
        return jsonify(historico)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
