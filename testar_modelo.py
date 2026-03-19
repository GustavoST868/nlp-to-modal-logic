import tkinter as tk
from tkinter import messagebox, scrolledtext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import threading
from configuracao import Configuracao

class InterfaceLogicaModal:
    def __init__(self, root):
        self.root = root
        self.root.title("Tradutor de Lógica Modal Premium")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f4f8")
        
        # load configuration/carregar configuração
        self.config = Configuracao()
        
        # colors and styles/cores e estilos
        self.PRIMARY_COLOR = "#2c3e50"
        self.ACCENT_COLOR = "#3498db"
        self.SUCCESS_COLOR = "#27ae60"
        self.ERROR_COLOR = "#e74c3c"
        self.BG_COLOR = "#f0f4f8"
        
        self.modelo = None
        self.tokenizador = None
        self.device = None
        
        self.criar_layout()
        
        # load model in a separate thread to avoid freezing the ui/carregar modelo em uma thread separada para não travar a ui
        self.status_label.config(text=f"Carregando modelo de {self.config.DIRETORIO_SAIDA}... Aguarde.", fg=self.ACCENT_COLOR)
        threading.Thread(target=self.inicializar_modelo, daemon=True).start()
    
    def inicializar_modelo(self):
        caminho_modelo = self.config.DIRETORIO_SAIDA
        try:
            if not os.path.exists(caminho_modelo) or not os.listdir(caminho_modelo):
                raise ValueError("Modelo não encontrado. Treine o modelo primeiro com treinar_modelo.py")
                
            self.tokenizador = AutoTokenizer.from_pretrained(caminho_modelo)
            self.modelo = AutoModelForSeq2SeqLM.from_pretrained(caminho_modelo)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.modelo.to(self.device)
            
            self.root.after(0, lambda: self.status_label.config(text="Modelo carregado com sucesso!", fg=self.SUCCESS_COLOR))
        except Exception as e:
            msg = f"Erro ao carregar modelo: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Erro", msg))
            self.root.after(0, lambda: self.status_label.config(text="Erro ao carregar modelo.", fg=self.ERROR_COLOR))
    
    def criar_layout(self):
        # main container/container principal
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # header/cabeçalho
        header = tk.Label(
            main_frame, 
            text="Tradutor de Linguagem Natural → Lógica Modal",
            font=("Helvetica", 18, "bold"),
            bg=self.BG_COLOR,
            fg=self.PRIMARY_COLOR
        )
        header.pack(pady=(0, 20))
        
        # input panel/painel de entrada
        input_frame = tk.LabelFrame(
            main_frame, 
            text=" Texto de Entrada ", 
            font=("Helvetica", 10, "bold"),
            bg=self.BG_COLOR,
            padx=10, 
            pady=10
        )
        input_frame.pack(fill=tk.X, pady=10)
        
        self.entrada_texto = tk.Entry(
            input_frame, 
            font=("Consolas", 12),
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.entrada_texto.pack(fill=tk.X, padx=5, pady=5)
        self.entrada_texto.bind("<Return>", lambda e: self.gerar_logica())
        
        # examples buttons/botões de exemplos
        exemplos_frame = tk.Frame(input_frame, bg=self.BG_COLOR)
        exemplos_frame.pack(fill=tk.X, pady=5)
        
        exemplos = [
            ("Pai", "a é pai de b"),
            ("Necessidade", "É necessário estudar"),
            ("Possibilidade", "Talvez chova amanhã"),
            ("Saber", "João sabe que Maria está em casa")
        ]
        
        for label, texto in exemplos:
            btn = tk.Button(
                exemplos_frame, 
                text=label, 
                command=lambda t=texto: self.usar_exemplo(t),
                font=("Helvetica", 9),
                bg="#ecf0f1",
                relief=tk.FLAT,
                padx=10
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # action buttons/botões de ação
        actions_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        actions_frame.pack(pady=10)
        
        self.btn_gerar = tk.Button(
            actions_frame, 
            text="GERAR TRADUÇÃO", 
            command=self.gerar_logica,
            font=("Helvetica", 12, "bold"),
            bg=self.ACCENT_COLOR,
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.btn_gerar.pack(side=tk.LEFT, padx=10)
        
        btn_limpar = tk.Button(
            actions_frame, 
            text="LIMPAR", 
            command=self.limpar_campos,
            font=("Helvetica", 12),
            bg="#95a5a6",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        btn_limpar.pack(side=tk.LEFT, padx=10)
        
        # output panel/painel de saída
        output_frame = tk.LabelFrame(
            main_frame, 
            text=" Tradução Gerada ", 
            font=("Helvetica", 10, "bold"),
            bg=self.BG_COLOR,
            padx=10, 
            pady=10
        )
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.saida_texto = scrolledtext.ScrolledText(
            output_frame, 
            font=("Consolas", 14),
            height=6,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.saida_texto.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # status bar/barra de status
        self.status_label = tk.Label(
            main_frame, 
            text="Pronto.", 
            font=("Helvetica", 10, "italic"),
            bg=self.BG_COLOR,
            fg=self.PRIMARY_COLOR
        )
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
            messagebox.showwarning("Aviso", "O modelo ainda não foi carregado.")
            return
            
        texto_entrada = self.entrada_texto.get().strip()
        if not texto_entrada:
            messagebox.showwarning("Aviso", "Por favor, digite algo para traduzir.")
            return
        
        self.status_label.config(text="Gerando tradução...", fg=self.ACCENT_COLOR)
        self.btn_gerar.config(state=tk.DISABLED)
        
        # run generation in the background to avoid freezing the ui/rodar geração em background para não travar a ui
        threading.Thread(target=self.tarefa_geracao, args=(texto_entrada,), daemon=True).start()
    
    def tarefa_geracao(self, texto):
        try:
            texto_com_prefixo = self.config.PREFIXO_TASK + texto
            ids_entrada = self.tokenizador.encode(texto_com_prefixo, return_tensors="pt").to(self.device)
            saidas = self.modelo.generate(
                ids_entrada, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            traducao = self.tokenizador.decode(saidas[0], skip_special_tokens=True)
            
            self.root.after(0, lambda: self.mostrar_resultado(traducao))
        except Exception as e:
            self.root.after(0, lambda: self.mostrar_erro(str(e)))
    
    def mostrar_resultado(self, resultado):
        self.saida_texto.delete(1.0, tk.END)
        self.saida_texto.insert(tk.END, resultado)
        self.status_label.config(text="Tradução concluída!", fg=self.SUCCESS_COLOR)
        self.btn_gerar.config(state=tk.NORMAL)
    
    def mostrar_erro(self, erro):
        messagebox.showerror("Erro na Geração", f"Ocorreu um erro: {erro}")
        self.status_label.config(text="Erro na tradução.", fg=self.ERROR_COLOR)
        self.btn_gerar.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceLogicaModal(root)
    root.mainloop()
