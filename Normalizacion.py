from bs4 import BeautifulSoup #permite obtener el texto de un archivo html
import re #expresiones regulares
import nltk
nltk.download('punkt') #tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy #lematizar
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("es_core_news_sm")
from collections import defaultdict
import numpy as np


def ObtainTextHTML(file_path):
    """
    obtiene el texto del HTML, dejando los puntos finales de cada oración
    Args: 
        file_path: dirección del archivo
    return: texto (string)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    # Obtener solo el texto del HTML (manteniendo los espacios)
    text = ' '.join(soup.stripped_strings)
    # Limpiar el texto para eliminar caracteres no deseados
    clean_text = re.sub(r'\s+', ' ', text.lower())  # Eliminar espacios adicionales
    text = re.sub(r'[^a-zA-ZñÑáéíóúÁÉÍÓÚ\s]', ' ', clean_text)  # Eliminar caracteres no alfanuméricos
    file.close()
    #print(text)
    return text

def Tokenization(text): #recibe un texto /expected a text
    """
    Tokeniza el texto
    Args:
        text: recibe texto
    Returns: lista de tokens

    """
    tokens =  nltk.word_tokenize(text, language="spanish")
#    print(tokens)
    #print("cantidad de tokens: ", len(tokens))
    return tokens

def StopWords(tokens): #recibe una lista /expected a list
    
    """
    Elimina StopWords
    Args:
        tokens: lista de tokens
    Returns: lista de tokens filtrados
    """
    stop_words = set(stopwords.words('spanish')) #lista de stopwords en español
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_tokens_after_re = [word for word in filtered_tokens if not re.match(r"html|htm|www|com|\d+|\s+|exe+\d\b|editorial|mx|excelsior|https|http|e990519_mod|\b[a-zA-Z]\b|aunque|excelsior|fin\d+\b|Miércoles|Mayo|art\d+\b|emod.htm", word, flags=re.IGNORECASE)]
    #print(filtered_tokens_after_re)
    return filtered_tokens_after_re

def get_post_tags(tokens, nlp):
    
    """
    Etiqueta cada palabra
    Args:
        tokens (list): lista de tokens
        nlp: permite obtener informacion del texto
    Returns: lista de etiquetas
    """
    text = ' '.join(tokens)
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    #print(pos_tags)
    return pos_tags
    

def Lemmatization(list_without_stopW, nlp, lemma_dict):#recibe una lista /expected a list    
    """
    covierte un texto en sus lemmas
    Args:
        list_without_stopWords (list):
        nlp : libreria spacy que permite lemmatizar, se debe declarar antes.
        lemma_dict: diccionario adicional del lemmas en caso de contar
    Returns: lista de lemmas
    """
    text = ' '.join(list_without_stopW)
    doc = nlp(text)
    #lemmas = [token.lemma_ for token in doc]
    custom_lemmas = {}
    for word, lemma in lemma_dict:
        custom_lemmas[word] = lemma
    
    lemmas = []
    for token in doc:
        if token.text in custom_lemmas:
            lemmas.append(custom_lemmas[token.text])
        else:
            lemmas.append(token.lemma_)
    #print(lemmas)
    return lemmas

def Counter(lemmas):
    """
    Cuenta la frecuencia de palabras en una lista
    Args:
        lemmas (list): lista de palabras
    Returns: diccionario de palabras y su frecuencia
    """
    word_counts = {}    
    for word in lemmas:    
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    #print(word_counts)
    return word_counts

def DividirCorpus(tokens): #debe de contener art## no eliminar de stopwords
    """
    divide el documento en articulos, (no se debe eliminar arg en las stopwords)
    Args:
        tokens: lista de tokens
    retunrs: lista de listas tokenizadas
    """
    patron_inicio_noticia = re.compile(r"www")
    indices_inicio_noticias = [i for i, palabra in enumerate(tokens) if patron_inicio_noticia.match(palabra)]
    noticias = []
    for i in range(len(indices_inicio_noticias)):
        inicio = indices_inicio_noticias[i]
        fin = indices_inicio_noticias[i + 1] if i + 1 < len(indices_inicio_noticias) else len(tokens)
        noticias.append(tokens[inicio:fin])
    #for i, noticia in enumerate(noticias, 1):
     #   print(f"Noticia {i}: {noticia}")
    return noticias

def Savetxt(file_path, info):
    with open(file_path, "w", encoding="utf-8") as text_file:
        text_file.write(info)
    
def SaveListTxt(file_path, info):
    with open(file_path, "w", encoding="utf-8") as text_file:
        text_file.write(" ".join(str(info)))    

def main():
    lemma_dict = [("abandonado", "abandonar"), ("abanderado", "abanderar"), ("distinguido", "distinguir"), ("retoma", "retomar"), ("rezagado", "rezagar"), ("especularmente", "especular"), ("manejo", "manejar"), ("destacado", "destacar"), ("inicien", "iniciar"), ("relevancia", "relevante"), ("ampliará", "ampliar"), ("expreso", "expres"), ("curiosamente", "curiosidad"), ("abrazado", "abrazar"), ("afiliada", "afiliar"), ("sancionado", "sancionar"), ("adquirido", "adquirir"), ("cafecito", "cafe"), ("manita", "mano"), ("maneja", "manejar")]
    file_path = r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\e990519_mod.htm" 
    text = ObtainTextHTML(file_path) #regresa un texto/ return text, recibe una ruta/expect path
    Savetxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\text.txt", text)
    tokens = Tokenization(text) # regresa una lista/return list, recibe un texto/expect text
    SaveListTxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\tokens.txt", tokens)
    stopwords = StopWords(tokens) # recibe una lista/expect list, regresa una lista/return list
    SaveListTxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\stopwords.txt", stopwords)
    lemmas = Lemmatization(stopwords, nlp, lemma_dict) #expect a path and list/ return set     
    SaveListTxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\lemmas.txt", lemmas)
    count = Counter(lemmas)
    SaveListTxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\count.txt", count)
    tags = get_post_tags(lemmas, nlp)
    SaveListTxt(r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\tags.txt", tags)
    corpus = DividirCorpus(lemmas)
    SaveListTxt(r'C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\Corpus.txt', corpus)    
if __name__ == "__main__":
    main()
