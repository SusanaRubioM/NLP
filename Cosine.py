from Normalizacion import Tokenization, Lemmatization, StopWords, ObtainTextHTML
import spacy
nlp = spacy.load("es_core_news_sm")
from BM25 import Vocabulary_CountTerms
from collections import defaultdict
import numpy as np

def Coocurrency_matrix(window, vocabulary, lemmas):
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    
    for i, lemma in enumerate(lemmas):
        start = max(0, i - window)
        end = min(len(lemmas), i + window + 1)
        
        for j in range(start, end):
            if j != i:  
                cooccurrence_counts[lemma][lemmas[j]] += 1
                
    vocabulary = sorted(cooccurrence_counts.keys())

    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)
    
    for i, word1 in enumerate(vocabulary):
        for j, word2 in enumerate(vocabulary):
            # Verificar si la palabra1 y la palabra2 son diferentes antes de contar la ocurrencia
            if word1 != word2:
                matrix[i, j] = cooccurrence_counts[word1][word2]
    
    return matrix, vocabulary

def find_word_position(word, vocabulary):
    """
    Encuentra la posición de una palabra específica en el vocabulario.
    vocabulary(list): debe estar ordenada
    """
    try:
        index = vocabulary.index(word)
        return index
    except ValueError:
        print("La palabra no está en el vocabulario.")
        return None
    
def CosineSimilarity(index, matrix_coo, vocabulary_sorted):
    """
    Calcula la similitud del coseno entre el vector de una palabra específica (en el índice dado) y todos los otros vectores en la matriz.
    """
    vectorWordA = matrix_coo[index]
    similarities = [(word, np.dot(vectorWordA, matrix_coo[i]) / (np.linalg.norm(vectorWordA) * np.linalg.norm(matrix_coo[i]))) for i, word in enumerate(vocabulary_sorted)]
    similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities_sorted




def main():
    file_path = r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\e990519_mod.htm" 
    lemma_dict = [("abandonado", "abandonar"), ("abanderado", "abanderar"), ("distinguido", "distinguir"), ("retoma", "retomar"), ("rezagado", "rezagar"), ("especularmente", "especular"), ("manejo", "manejar"), ("destacado", "destacar"), ("inicien", "iniciar"), ("relevancia", "relevante"), ("ampliará", "ampliar"), ("expreso", "expres"), ("curiosamente", "curiosidad"), ("abrazado", "abrazar"), ("afiliada", "afiliar"), ("sancionado", "sancionar"), ("adquirido", "adquirir"), ("cafecito", "cafe"), ("manita", "mano"), ("maneja", "manejar")]
    Word = "crecimiento"
    text = ObtainTextHTML(file_path)
    tokens = Tokenization(text)
    tokens_Without_SW = StopWords(tokens)
    lemmas = Lemmatization(tokens_Without_SW, nlp, lemma_dict)
    vocabulary, dl = Vocabulary_CountTerms(lemmas)
    matrix_coo, vocabulary_sorted = Coocurrency_matrix(2, vocabulary, lemmas)
    index = find_word_position("crecimiento", vocabulary_sorted)
    similarity = CosineSimilarity(index, matrix_coo, vocabulary_sorted)
    print(similarity)


if __name__ == "__main__":
    main()