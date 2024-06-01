from Normalizacion import ObtainTextHTML, Tokenization, StopWords, Lemmatization
import spacy
nlp = spacy.load("es_core_news_sm")

def Vocabulary_CountTerms(lemmas):
    """
    |dl| longitud del documento, es decir el numero total de terminos
    Args:
        lemmas
    return: vocabulario(list) y entero
    """
    vocabulary = []
    for word in lemmas:
        if word not in vocabulary:
            vocabulary.append(word)
    return vocabulary, len(vocabulary)    

def Ocurrence_Frequency(Wi, lemmas):
    """
    C(Wi,dl) es la frecuencia de ocurrencia del término wi(termino especifico) en el documento dl
    Args:
        Wi(str): termino especifico
        lemmas(list): lista de palabras
    return: int"""
    C=0
    for word in lemmas:
        if word == Wi:
            C +=1
    return C

def calcular_longitud_promedio(documento):
    total_terminos = len(documento.split())
    return total_terminos

def BM25(Wi, C, dl, k, b, avdl):
    """
    C(wi,dl)(int):frecuencia de ocurrencia del termino
    k y b son parametros de ajuste
    |dl| longitud del documento
    avdl longitud promedio del documento
    """
    bm25 = ((k + 1)*C)/(C + k * (1 - b + b *(dl/avdl)))
    return bm25

def main():
    file_path = r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\e990519_mod.htm" 
    lemma_dict = [("abandonado", "abandonar"), ("abanderado", "abanderar"), ("distinguido", "distinguir"), ("retoma", "retomar"), ("rezagado", "rezagar"), ("especularmente", "especular"), ("manejo", "manejar"), ("destacado", "destacar"), ("inicien", "iniciar"), ("relevancia", "relevante"), ("ampliará", "ampliar"), ("expreso", "expres"), ("curiosamente", "curiosidad"), ("abrazado", "abrazar"), ("afiliada", "afiliar"), ("sancionado", "sancionar"), ("adquirido", "adquirir"), ("cafecito", "cafe"), ("manita", "mano"), ("maneja", "manejar")]
    Wi = "crecimiento"
    k = 1.2
    b = 0.75
    text = ObtainTextHTML(file_path)
    tokens = Tokenization(text)
    tokens_Without_SW = StopWords(tokens)
    lemmas = Lemmatization(tokens_Without_SW, nlp, lemma_dict)
    vocabulary, dl = Vocabulary_CountTerms(lemmas)
    C = Ocurrence_Frequency(Wi, lemmas)
    bm25 = BM25(Wi, C, dl, k, b, avdl=1)
    print("Relevancia en el documento de \"Crecimiento\": ", bm25)
    

if __name__ == "__main__":
    main()