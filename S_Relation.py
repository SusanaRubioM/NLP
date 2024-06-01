from Normalizacion import StopWords, Lemmatization, Tokenization
import spacy
from collections import defaultdict
from bs4 import BeautifulSoup #permite obtener el texto de un archivo html
import re #expresiones regulares

nlp = spacy.load("es_core_news_sm")


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
    text = re.sub(r'[^a-zA-ZñÑáéíóúÁÉÍÓÚ\s.]', '', clean_text)  # Eliminar caracteres no alfanuméricos
    file.close()
    #print(text)
    return text

def Token_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def main():
    file_path = r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\e990519_mod.htm" 
    lemma_dict = [("abandonado", "abandonar"), ("abanderado", "abanderar"), ("distinguido", "distinguir"), ("retoma", "retomar"), ("rezagado", "rezagar"), ("especularmente", "especular"), ("manejo", "manejar"), ("destacado", "destacar"), ("inicien", "iniciar"), ("relevancia", "relevante"), ("ampliará", "ampliar"), ("expreso", "expres"), ("curiosamente", "curiosidad"), ("abrazado", "abrazar"), ("afiliada", "afiliar"), ("sancionado", "sancionar"), ("adquirido", "adquirir"), ("cafecito", "cafe"), ("manita", "mano"), ("maneja", "manejar")]
    text = ObtainTextHTML(file_path)
    sentences = Token_sentences(text)
    vocabulary = set()
    vocabulary_count = defaultdict(int)
    bigram_count = defaultdict(int)
    total_words = 0

    for sentence in sentences:
        sentence_tokens = Tokenization(sentence)
        sentence_without_stopwords = StopWords(sentence_tokens)
        sentence_lemma = Lemmatization(sentence_without_stopwords, nlp, lemma_dict)
        unique_words = set(sentence_lemma)
        vocabulary.update(unique_words)
        for word in unique_words:
            vocabulary_count[word] += 1
        for i in range(len(sentence_lemma) - 1):
            bigram = (sentence_lemma[i], sentence_lemma[i + 1])
            bigram_count[bigram] += 1
        total_words += len(sentence_lemma)

    #print("Palabra - Cantidad de oraciones en las que aparece:")
    #for word, count in sorted(vocabulary_count.items()):
    #    print(f"{word}: {count}")

    w_star = "crecimiento"  # Palabra específica

    print(f"Suavizado de Laplace para la palabra {w_star} (H en orden descendente, MI en orden ascendente):")
    
    # Calcular H y MI para cada palabra en el vocabulario
    results_h = []
    results_mi = []
    for word in vocabulary:
        h = (bigram_count[(word, w_star)] + 1) / (vocabulary_count[word] + len(vocabulary))
        mi = bigram_count[(word, w_star)] * total_words / (vocabulary_count[word] * vocabulary_count[w_star])
        results_h.append((word, h))
        results_mi.append((word, mi))

    # Ordenar y imprimir H en orden descendente
    print("\nEntropia condicional h(w*|w) en orden descendente:")
    for word, h_value in sorted(results_h, key=lambda x: x[1], reverse=True):
        print(f"{word}: {h_value}")

    # Ordenar y imprimir MI en orden ascendente
    print("\nInformación mutua mi(w*, w) en orden ascendente:")
    for word, mi_value in sorted(results_mi, key=lambda x: x[1], reverse=True):
        print(f"{word}: {mi_value}")

if __name__ == "__main__":
    main()
