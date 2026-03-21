import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def baixar_recursos():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('rslp', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

baixar_recursos()

class PreProcessador:
    def __init__(self):
        CONECTIVOS_LOGICOS = {
            'e', 'ou', 'não', 'se', 'então', 'entao', 'nem', 'mas', 
            'cada', 'todo', 'toda', 'todos', 'todas', 'algum', 'alguma', 
            'alguns', 'algumas', 'existe', 'posso', 'pode'
        }
        
        try:
            sw_padrao = set(stopwords.words('portuguese'))
            self.stop_words = set()
            for palavra in sw_padrao:
                if palavra not in CONECTIVOS_LOGICOS:
                    self.stop_words.add(palavra)
            
            self.stemmer = RSLPStemmer()
        except Exception:
            self.stop_words = set()
            self.stemmer = None
            
        self.vectorizer = TfidfVectorizer()

    def limpar_texto(self, texto):
        if not isinstance(texto, str):
            return ""
            
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        
        entrada_split = texto.split()
        tokens = []
        
        for t in entrada_split:
            if t not in self.stop_words:
                if self.stemmer:
                    tokens.append(self.stemmer.stem(t))
                else:
                    tokens.append(t)
            
        resultado = " ".join(tokens)
        return resultado

    def processar_dataset(self, textos):
        processados = []
        for t in textos:
            limpo = self.limpar_texto(t)
            processados.append(limpo)
        return processados

    def aplicar_tfidf(self, textos, fit=True):
        if fit:
            return self.vectorizer.fit_transform(textos)
        return self.vectorizer.transform(textos)

    def obter_termos(self):
        return self.vectorizer.get_feature_names_out()

if __name__ == "__main__":
    preprocessador = PreProcessador()
    frases = [
        "a é pai de b",
        "toda pessoa é uma pessoa."
    ]
    
    frases_limpas = preprocessador.processar_dataset(frases)
    
    print("Frases originais:", frases)
    print("Frases limpas:", frases_limpas)
    
    matriz_tfidf = preprocessador.aplicar_tfidf(frases_limpas)
    print("Formato da matriz TF-IDF:", matriz_tfidf.shape)
    print("Alguns tokens aprendidos:", preprocessador.obter_termos()[:10])
