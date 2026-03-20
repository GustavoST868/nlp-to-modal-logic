import tkinter as tk
from tkinter import messagebox, scrolledtext, Toplevel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from configuracao import Configuracao

class InterfaceLogicaModal:
    """
    Interface gráfica utilizando Tkinter para tradução de linguagem natural para lógica modal em tempo real e visualização de desempenho.
    Graphical interface using Tkinter for natural language to modal logic translation in real time and performance visualization.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Tradutor de Lógica Modal Premium")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f4f8")
        
        self.config = Configuracao()
        
        self.PRIMARY_COLOR = "#2c3e50"
        self.ACCENT_COLOR = "#3498db"
        self.SUCCESS_COLOR = "#27ae60"
        self.ERROR_COLOR = "#e74c3c"
        self.BG_COLOR = "#f0f4f8"
        
        self.modelo = None
        self.tokenizador = None
        self.device = None
        
        self.criar_layout()
        
        self.status_label.config(text=f"Carregando modelo de {self.config.DIRETORIO_SAIDA}... Aguarde.", fg=self.ACCENT_COLOR)
        threading.Thread(target=self.inicializar_modelo, daemon=True).start()
    
    def inicializar_modelo(self):
        """
        Carrega o modelo treinado e o tokenizador.
        Loads the trained model and tokenizer.
        """
        caminho_modelo = self.config.DIRETORIO_SAIDA
        try:
            if not os.path.exists(caminho_modelo) or not os.listdir(caminho_modelo):
                raise ValueError("Modelo não encontrado. Treine o modelo primeiro.")
                
            self.tokenizador = AutoTokenizer.from_pretrained(caminho_modelo, use_fast=False)
            self.modelo = AutoModelForSeq2SeqLM.from_pretrained(caminho_modelo)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo.to(self.device)
            
            self.root.after(0, lambda: self.status_label.config(text="Modelo carregado com sucesso!", fg=self.SUCCESS_COLOR))
        except Exception as e:
            msg = f"Erro ao carregar modelo: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text="Erro ao carregar modelo.", fg=self.ERROR_COLOR))
    
    def criar_layout(self):
        """
        Gera os elementos visuais da interface.
        Generates visual elements of the interface.
        """
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header = tk.Label(
            main_frame, 
            text="Tradutor de Linguagem Natural → Lógica Modal",
            font=("Helvetica", 18, "bold"),
            bg=self.BG_COLOR, fg=self.PRIMARY_COLOR
        )
        header.pack(pady=(0, 20))
        
        input_frame = tk.LabelFrame(main_frame, text=" Texto de Entrada ", font=("Helvetica", 10, "bold"), bg=self.BG_COLOR, padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        self.entrada_texto = tk.Entry(input_frame, font=("Consolas", 12), relief=tk.FLAT, highlightthickness=1, highlightbackground="#bdc3c7")
        self.entrada_texto.pack(fill=tk.X, padx=5, pady=5)
        self.entrada_texto.bind("<Return>", lambda e: self.gerar_logica())
        
        exemplos_frame = tk.Frame(input_frame, bg=self.BG_COLOR)
        exemplos_frame.pack(fill=tk.X, pady=5)
        
        exemplos = [("Pai", "a é pai de b"), ("Necessidade", "É necessário estudar"), ("Possibilidade", "Talvez chova amanhã"), ("Saber", "João sabe que Maria está em casa")]
        for label, texto in exemplos:
            tk.Button(exemplos_frame, text=label, command=lambda t=texto: self.usar_exemplo(t), font=("Helvetica", 9), bg="#ecf0f1", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=2)
        
        actions_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        actions_frame.pack(pady=10)
        
        self.btn_gerar = tk.Button(actions_frame, text="GERAR TRADUÇÃO", command=self.gerar_logica, font=("Helvetica", 12, "bold"), bg=self.ACCENT_COLOR, fg="white", relief=tk.FLAT, padx=15, pady=8, cursor="hand2")
        self.btn_gerar.pack(side=tk.LEFT, padx=5)
        
        tk.Button(actions_frame, text="LIMPAR", command=self.limpar_campos, font=("Helvetica", 12), bg="#95a5a6", fg="white", relief=tk.FLAT, padx=15, pady=8, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        tk.Button(actions_frame, text="VER DESEMPENHO", command=self.exibir_desempenho, font=("Helvetica", 12, "bold"), bg=self.SUCCESS_COLOR, fg="white", relief=tk.FLAT, padx=15, pady=8, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        output_frame = tk.LabelFrame(main_frame, text=" Tradução Gerada ", font=("Helvetica", 10, "bold"), bg=self.BG_COLOR, padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.saida_texto = scrolledtext.ScrolledText(output_frame, font=("Consolas", 14), height=6, relief=tk.FLAT, highlightthickness=1, highlightbackground="#bdc3c7")
        self.saida_texto.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_label = tk.Label(main_frame, text="Pronto.", font=("Helvetica", 10, "italic"), bg=self.BG_COLOR, fg=self.PRIMARY_COLOR)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def usar_exemplo(self, texto):
        self.entrada_texto.delete(0, tk.END)
        self.entrada_texto.insert(0, texto)
    
    def limpar_campos(self):
        self.entrada_texto.delete(0, tk.END)
        self.saida_texto.delete(1.0, tk.END)
        self.status_label.config(text="Campos limpos.", fg=self.PRIMARY_COLOR)
    
    def gerar_logica(self):
        if not self.modelo:
            messagebox.showwarning("Aviso", "Modelo não carregado.")
            return
        t = self.entrada_texto.get().strip()
        if not t: return
        self.status_label.config(text="Gerando tradução...", fg=self.ACCENT_COLOR)
        threading.Thread(target=self.tarefa_geracao, args=(t,), daemon=True).start()
    
    def tarefa_geracao(self, texto):
        try:
            inp = self.tokenizador(self.config.PREFIXO_TASK + texto, return_tensors="pt").to(self.device)
            out = self.modelo.generate(**inp, max_length=128, num_beams=4)
            trad = self.tokenizador.decode(out[0], skip_special_tokens=True)
            self.root.after(0, lambda: self.mostrar_resultado(trad))
        except Exception as e:
            self.root.after(0, lambda: self.mostrar_erro(str(e)))
    
    def mostrar_resultado(self, res):
        self.saida_texto.delete(1.0, tk.END)
        self.saida_texto.insert(tk.END, res)
        self.status_label.config(text="Concluído!", fg=self.SUCCESS_COLOR)
    
    def mostrar_erro(self, err):
        messagebox.showerror("Erro", err)
        self.status_label.config(text="Erro.", fg=self.ERROR_COLOR)

    def exibir_desempenho(self):
        """
        Abre uma nova janela para exibir os gráficos e métricas de desempenho do último treinamento.
        Opens a new window to display performance graphs and metrics from the last training.
        """
        caminho_hist = os.path.join(self.config.DIRETORIO_SAIDA, "historico_treinamento.json")
        if not os.path.exists(caminho_hist):
            messagebox.showinfo("Informação", "Histórico de treinamento não encontrado. Treine o modelo primeiro.")
            return
            
        with open(caminho_hist, "r") as f:
            historico = json.load(f)
            
        janela_perf = Toplevel(self.root)
        janela_perf.title("Métricas de Desempenho")
        janela_perf.geometry("800x650")
        janela_perf.configure(bg="white")
        
        epochs = []
        train_loss = []
        eval_loss = []
        eval_rouge1 = []
        
        for entry in historico:
            if "loss" in entry and "epoch" in entry:
                epochs.append(entry["epoch"])
                train_loss.append(entry["loss"])
            if "eval_loss" in entry and "epoch" in entry:
                eval_loss.append(entry["eval_loss"])
                if "eval_rouge1" in entry:
                    eval_rouge1.append(entry["eval_rouge1"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        fig.tight_layout(pad=5.0)
        
        ax1.plot(train_loss, label="Train Loss", color="blue")
        if eval_loss:
            ax1.plot(eval_loss, label="Eval Loss", color="red")
        ax1.set_title("Curva de Perda (Loss)")
        ax1.set_xlabel("Passos/Épocas")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        if eval_rouge1:
            ax2.plot(eval_rouge1, label="ROUGE-1", color="green")
            ax2.set_title("Evolução da Métrica ROUGE-1")
            ax2.set_xlabel("Épocas")
            ax2.set_ylabel("Score")
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Métricas de avaliação não disponíveis", ha='center')
            
        canvas = FigureCanvasTkAgg(fig, master=janela_perf)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceLogicaModal(root)
    root.mainloop()
