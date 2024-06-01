from Normalizacion import Tokenization, DividirCorpus, ObtainTextHTML, Lemmatization
from nltk.corpus import stopwords
import re
import random
from BM25 import Vocabulary_CountTerms, Ocurrence_Frequency
import numpy as np
import nltk
from nltk.probability import FreqDist
import spacy
nlp = spacy.load("es_core_news_sm")

def StopWordss(tokens):
    """
    Elimina StopWords y otros caracteres no deseados.
    Args:
        tokens: lista de tokens
    Returns: lista de tokens filtrados
    """
    stop_words = set(stopwords.words('spanish'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_tokens_after_re = [word for word in filtered_tokens if not re.match(r"aar|html|e|miércoles|mod|e990519_mod.htm|htm|com|\d+|\s+|exe+\d\b|editorial|mx|excelsior|https|http|e990519_mod|\b[a-zA-Z]\b|excelsior|fin\d+\b|emod.htm|emodhtm|pues", word, flags=re.IGNORECASE)]
    return filtered_tokens_after_re

def E_Step(Prob_Background, Prob_Topic, Background_Word_Probability, Topic_Word_Probability):
    """
    Calcula la responsabilidad de cada palabra en el corpus, que es una medida de cuán probable es que una palabra provenga del tema en lugar del fondo (background)
    args:
        Prob_Background & Prob_Topic (float): a priori probabilities
        Background_Word_Probability & Topic_Word_Probability (array): probabilities to observe each word in document d(topics) and b(stopwords)
    returns:
        PzW ():
    """
    PzW = (Prob_Topic * Topic_Word_Probability) / ((Prob_Topic * Topic_Word_Probability) + (Prob_Background * Background_Word_Probability))
    return PzW

def M_Step(Counts, PzW):
    """
    Actualiza las probabilidades de las palabras dentro del tema basado en las responsabilidades calculadas en el E-Step.
    
    args:
        Counts(list int): frequency for each word in the vocabulary for each article
        PzW(list float): How likely w is from 0d
    returns:
        Topic_word_prob (list): p(w|d) used for the next iteration of E_step
    """
    Topic_word_prob = (Counts * PzW) / (np.sum(Counts * PzW))
    return Topic_word_prob


def Compute_Document_Likelihood(background_word_probs, topic_word_probs, counts, prob_background, prob_topic, num_iterations=50):
    """
    Calcula la verosimilitud del documento.

    Args:
    - Background_word_probs: Probabilidades de ocurrencia de cada palabra en el fondo (background).
    - Prob_Background: Probabilidad a priori del fondo.
    - Prob_Topic: Probabilidad a priori del tema.
    - Iterations: Número de iteraciones.
    - Topic_Word_Probs: Probabilidades de ocurrencia de cada palabra en el tema.
    - Counts: Frecuencias de cada palabra en el corpus (para un solo artículo).
    - print_likelihood: Booleano que indica si imprimir o no la verosimilitud del documento.

    Return:
    - document_likelihood: Verosimilitud del documento.
    """    
    arguments_for_logarithm = background_word_probs * prob_background + topic_word_probs * prob_topic
    logarithms = np.log(arguments_for_logarithm)
    for i in range(len(logarithms)):
        product = logarithms[i] * counts[i]
        logarithms[i] = product
    
    document_likelihood = np.sum(logarithms)
    document_likelihood = round(document_likelihood, 5)
    return document_likelihood
    
def Obtain_Top_Words(Vocabulary, N_top, Probabilities):
    List_VocProb = None
    List_VocProb = []
    for i in range (len(Vocabulary)):
        aux = (Vocabulary[i] , Probabilities[i])
        List_VocProb.append(aux)
    List_VocProb = sorted(List_VocProb, key=lambda x: x[1], reverse=True)
    print (List_VocProb[:N_top])
    

def main():
    file_path = r"C:\Users\Susan\OneDrive\Documentos\T_Lenguaje_Natural\TLN-24\e990519_mod.htm" 
    lemma_dict = [("abandonado", "abandonar"),("abandona", "abandonar"), ("abandonaba", "abandonar"), ("abanderado", "abanderar"), ("distinguido", "distinguir"), ("retoma", "retomar"), ("rezagado", "rezagar"), ("especularmente", "especular"), ("manejo", "manejar"), ("destacado", "destacar"), ("inicien", "iniciar"), ("relevancia", "relevante"), ("ampliará", "ampliar"), ("expreso", "expres"), ("curiosamente", "curiosidad"), ("abrazado", "abrazar"), ("afiliada", "afiliar"), ("sancionado", "sancionar"), ("adquirido", "adquirir"), ("cafecito", "cafe"), ("manita", "mano"), ("maneja", "manejar")]
    text = ObtainTextHTML(file_path) # Regresa un texto
    tokens = Tokenization(text)
    TextClean = StopWordss(tokens) # Eliminar números e instrucciones HTML
    TextLematiced = Lemmatization(TextClean, nlp, lemma_dict)
    Text_divide = DividirCorpus(TextLematiced)
    fd = nltk.FreqDist(TextLematiced)
    vocabulary = sorted(list(fd.keys()))
    #print(vocabulary)
    fd = dict(fd)
    print(f"There are {len(Text_divide)} articles")
    print(f"The vocabulary has {len(vocabulary)} words")
    Background_Word = [Ocurrence_Frequency(word0, TextLematiced) for word0 in vocabulary]
    Sum = np.sum(Background_Word)
    Background_Word_Probability = Background_Word / Sum

    Prob_Background = 0.1
    Prob_Topic = 0.9
    Count_All_Articles = []
    for article in Text_divide: #frequency of each word in whole vocabulary, but in each article
        Count = [Ocurrence_Frequency(word, article) if Ocurrence_Frequency(word, article) > 0 else 0.00001 for word in vocabulary]
        Count_All_Articles.append(Count)
        

    for idx, article_counts in enumerate(Count_All_Articles):
        print(f"\nArticle {idx + 1}")
        num_iterations = 200
        prev_likelihood = 0
        document_likelihood_new = 0
        uncganged_likelihood_count = 0
        Probabilities_Initial = 1 / len(vocabulary)
        Topic_Word_Probability = np.full(shape=len(vocabulary), fill_value=Probabilities_Initial)
        for iteration in range(num_iterations):
            # E-Step
            PzW = E_Step(Prob_Background, Prob_Topic, Background_Word_Probability, Topic_Word_Probability)
            
            # M-Step
            new_Topic_Word_Probabilitys = M_Step(article_counts, PzW)
            #print(new_Topic_Word_Probabilitys)
            # Normalizar las nuevas probabilidades del tema
            Topic_Word_Probability = new_Topic_Word_Probabilitys / np.sum(new_Topic_Word_Probabilitys)
            #print(Topic_Word_Probability)
            #print(Background_Word_Probability[idx])
            document_likelihood = Compute_Document_Likelihood(Background_Word_Probability, Topic_Word_Probability, article_counts, Prob_Background, Prob_Topic, num_iterations=iteration)
            if document_likelihood == prev_likelihood:
                unchanged_likelihood_count += 1 
                if unchanged_likelihood_count >= 2:  
                    print(f'Iteration {iteration + 1}: Document likelihood is {document_likelihood}')
                    break
            else:
                unchanged_likelihood_count = 0  
            print(f'Iteration {iteration + 1}: Document likelihood is {document_likelihood}')
            prev_likelihood = document_likelihood

        print("\tTopic Words:")
        #print(Topic_Word_Probability)
        Obtain_Top_Words(vocabulary, 10 ,Topic_Word_Probability)
        print("\tBackground Words:")
        #print(Background_Word_Probability)
        Obtain_Top_Words(vocabulary, 10 ,Background_Word_Probability)


if __name__ == "__main__":
    main()

