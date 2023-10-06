#!/usr/bin/env python
# coding: utf-8

# # Importazione

# ## Importazione pacchetti

# In[1]:


import pandas as pd
import json
import re
import numpy as np
from collections import Counter
from functools import reduce
#from  tqdm import tqdm
import math
from nltk.stem import SnowballStemmer


# ## Import data

# In[2]:


df=pd.read_csv("Dati/light_dati_puliti.csv")
df.drop(columns={'Unnamed: 0'},inplace=True)

DF2=pd.read_csv("Dati/light_tfidf_index.csv", index_col=None)
DF2.drop(columns={'Unnamed: 0'},inplace=True)

with open("/Users/mattia/Desktop/Camera cafe/_SearchEngine/Dati/lista_gueststar.txt", 'r') as v:
    n=v.read()
star=n.split(",")


# ## Import vocabulary

# In[3]:


with open("Dati/light_vocabulary.json", 'r') as v:
    vocabulary = json.load(v)
    
with open("Dati/light_vocabulary.json", 'r') as wd:
    word_dict  = json.load(wd)
    
with open("Dati/light_inverted_idx.json", 'r') as inv_idx:
    inverted_idx = json.load(inv_idx)


# # Engine

# ## Stemmer

# In[4]:


def stem_text_light(txt):
    stemmer_snowball = SnowballStemmer('italian')
    txt=txt.lower()  #testo minuscolo
    result = re.sub(r'[^\w\s]', '', txt) #rimuovere punteggiatura

    doc=result.split() #lista delle parole
    articoli=["il","lo","la","i","gli","le", #articoli determinativi
              "un","uno","una",              #articoli indeterminativi
              "del"," dello", "della", "dei", "degli", "delle"] #articoli partitivi

    congiunzioni = ['e', 'o', 'ma', 'perché', 'poiché', 'anche se', 'benché', 'sebbene', 'quando', 'mentre', 'finché', 'affinché', 'se', 'così', 'cioè', 'anche', 'inoltre', 'oltre a', 'anzi', 'perciò', 'dunque', 'pertanto', 'quindi', 'oppure', 'altrimenti', 'neppure', 'sia... sia...', 'non solo... ma anche...', 'o... o...', 'se... se...', 'neanche... né...', 'per quanto', 'anche se', 'in quanto', 'come', 'cosicché', 'affinché', 'giacché', 'laddove', 'qualora', 'a meno che', 'affinché', 'prima che', 'benché', 'quanto', 'sebbene']


    preposizioni = ["di","a","da","in","con","su","per","tra","fra",           #preposizioni semplici
              'del', 'dello', 'della', 'dell’', 'dei', 'degli', 'delle', #preposizioni articolate
              'al', 'allo', 'alla', 'all’', 'ai', 'agli', 'alle',        #preposizioni articolate
              'dal', 'dallo', 'dalla', 'dall’', 'dai', 'dagli', 'dalle', #preposizioni articolate
              'nel', 'nello', 'nella', 'nell’', 'nei', 'negli', 'nelle', #preposizioni articolate
              'col', 'collo', 'colla', 'con l’', 'coi', 'cogli', 'colle',#preposizioni articolate
              'sul', 'sullo', 'sulla', 'sull’', 'sui', 'sugli', 'sulle']
    congiunzioni_coordinazione_italiane = [
        'e', 'o', 'ma', 'perché', 'mentre', 'se', 'quando', 'finché', 'come', 'oppure', 
        'anche', 'oltre a', 'così', 'quindi', 'perciò', 'inoltre', 'né', 'anzi', 
        'altrimenti', 'sia... sia...', 'o... o...', 'se... se...', 'neanche... né...', 'benché'
    ]

    determinanti_italiani = [
        'il', 'la', 'i', 'le', 'un', 'uno', 'una', 'dei', 'delle', 'degli', 'dell\'', 
        'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle', 
        'mia', 'mio', 'miei', 'mie', 'tua', 'tuo', 'tue', 'sua', 'suo', 'sue', 'nostra', 
        'nostro', 'nostre', 'vostra', 'vostro', 'vostre', 'loro', 'loro', 'loro', 
        'alcune', 'alcuni', 'alcuna', 'alcune', 'nessuno', 'nessuna', 'nessuni', 'nessune'
    ]

    #return [stemmer_snowball.stem(token) for token in doc if token not in articoli and token not in preposizioni and token not in congiunzioni]
    return [stemmer_snowball.stem(token) for token in doc if token not in determinanti_italiani and token not in congiunzioni_coordinazione_italiane]


# ## Funzione tdidf

# In[5]:


def querty_td_idf(query_str):
    #Inizialize dataset
    td_inf={}
    
    #Remove double spaces
    query_str=re.sub("\s{2,}"," ",query_str)
    
    #Stem the query
    stemmed_query=stem_text_light(query_str)
    
    
    word_in_query=set(stemmed_query)
    
    
    for word in word_in_query:
        #DF document frequency
        df_t=vocabulary[word]
        if df_t==0:
            raise Exception("Can't find a document that contain all the word in the query, check input for misspelled words.")
           # raise Value Error("Can't find a document that contain all the word in the query, check input for misspelled words.")
            #print("Can't find a document that contain all the word in the query, check input for misspelled words.")
            #return None
        else:
    
            #TF
            ##occurence of t word in the query
            t_in_document=" ".join(stemmed_query).count(word) #count on total query or stemmed query? for now stemmed query


            ##Word in  the query
            parole_in_d=len(stemmed_query) #count on stemmed query or total query?for now stemmed query

            tf=t_in_document/parole_in_d
            
            #DF document frequency
            #df_t=vocabulary[word]
            results = [inverted_idx[str(item)] for item in [word_dict[word] for word in stemmed_query if word in vocabulary]]
            df_t=len(results)

            #Inverse document frequency
            #N=len(df)
            N=7200
            idf=math.log(N/(df_t+1))


            #TD-INF
            td_inf_i=tf* idf

            td_inf[word]=td_inf_i  #dictionary of td_idf for every word in the query

    dataset=pd.DataFrame([td_inf])
    return dataset


# ## Ranking engine all Match
# Un documento deve contenere tutte le parole presenti nella query per essere ritrovato

# ### Funzioni

# In[6]:


def query_ranking_allMatch(query_str,k,season=None):
    t=""
    for i in set(query_str.split()):
        t += i + " "
    query_str=t.strip()

    query_str=re.sub("\s{2,}"," ",query_str)
    
    parole=[word for word in stem_text_light(query_str)]
    check_parole=[word for word in stem_text_light(query_str) if word in vocabulary]
    if len(parole)== len(check_parole):
        

        indici=list(DF2[stem_text_light(query_str)][(DF2[stem_text_light(query_str)].T != 0).all()].index)




        #Inizialize a list for store the value of cosine_similarity
        list_cosine_similarity=[]

        #Define a new dataset. In this dataset we will add the cosine similarity.
        data=df[df.index.isin(indici)][['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']]


        #Define a new dataset. In this dataset we will put the tf-idf of the matching documents.
        a=DF2[DF2.index.isin(indici)][stem_text_light(query_str)]
        a=a.reindex(sorted(a.columns), axis=1) 

        #Evaluate tf-idf of query
        td_idf_query=querty_td_idf(query_str)
        td_idf_query=td_idf_query.reindex(sorted(a.columns), axis=1) 



        #Evaluate cosine similarity between each document and the query.


        for i,row in a.iterrows():


            num=np.dot(td_idf_query.values.flatten(),row.values)
            den=np.linalg.norm(td_idf_query.values.flatten()) * np.linalg.norm(row.values)

            cosine=num/den
            list_cosine_similarity.append(cosine)

        data["Similarity"]= list_cosine_similarity
        dd=data.sort_values("Similarity",ascending=False)
        if season==None:
            dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']]
        else:
            dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']][df.season==season]
        
        #return dd
        if dd.shape[0]==0:
            print("Nessuna puntata ritrovata: \n   - Provare ad ampliare la ricerca (es: eliminare restrizioni sulle stagioni);\n   - Controllare ciò che è stato scritto.")
            return None
    
        dd.reset_index(drop=True,inplace=True)
        #display(HTML(dd.head(k).to_html()))
        return dd
    else:
        print("Impossibile trovare un documento che contenga tutta la parola nella query, controllare l'input per le parole errate.")
        return None


# ## Ranking engine
# Vengono ritornati i primi k documenti più rilevanti

# ### Funzione per le guest star

# In[7]:


def filter_dataframe_by_columns(df, column_names):
    condition = df[column_names[0]] == 1
    for i in range(1, len(column_names)):
        condition |= df[column_names[i]] == 1
    
    filtered_df = df[condition]
    return filtered_df


# ### Search only for season

# In[8]:


def only_season(season):
    return df[df.season==season]


# ### Search only guest star 

# In[9]:


def only_star(stars,season=None):
    data=filter_dataframe_by_columns(df,stars)
    if season != None:
        data=data[df.season==season]
    
    if data.shape[0]==0: #in caso il dataframe di output è vuoto si print un messaggio di errore
        print("Nessuna puntata ritrovata: \n   - Provare ad ampliare la ricerca (es: eliminare restrizioni sulle stagioni);\n   - Controllare ciò che è stato scritto.")
        return None
    return data


# ### Funzione di principale

# In[10]:


def query_ranking(query_str,k,season=None,stars=None,research_type=0):
    """
    season [None,1,2,3,4,5,6]
        Questo parametro permette di fare una scrematura delle puntate ritrovate dalla search engine in base alla stagione.
        
        - None -> non verrà fatta alcuna selezione sulle stagioni
        - 1 -> solo puntate della stagione 1
        - 2 -> solo puntate della stagione 2
        - 3 -> solo puntate della stagione 3
        - 4 -> solo puntate della stagione 4
        - 5 -> solo puntate della stagione 5
        - 6 -> solo puntate della stagione 6
        
        

    stars [None,lista delle star]
        Questo parametro permette di fare una scrematura delle puntate ritrovate dalla search engine in base alle guest star presenti.
        None -> non sarà fatta una ulteriore scrematura sulle guest star
        lista delle star -> una lista python contenente le star che si vogliono ricercare
    """
 
    
    stemmer_snowball = SnowballStemmer('italian')
    t=""
    for i in set(query_str.split()):
        t += i + " "
    query_str=t.strip()

    query_str=re.sub("\s{2,}"," ",query_str)

    parole=[word for word in stem_text_light(query_str)]
    check_parole=[word for word in stem_text_light(query_str) if word in vocabulary]


    indici=list(DF2[stem_text_light(query_str)].index)




    #Inizialize a list for store the value of cosine_similarity
    list_cosine_similarity=[]



    #Define a new dataset. In this dataset we will add the cosine similarity.
    data=df[df.index.isin(indici)]#[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']]

    


    #Define a new dataset. In this dataset we will put the tf-idf of the matching documents.
    a=DF2[DF2.index.isin(indici)][stem_text_light(query_str)]
    a=a.reindex(sorted(a.columns), axis=1) 

    #Evaluate tf-idf of query
    query_str=stemmer_snowball.stem(query_str)
    td_idf_query=querty_td_idf(query_str)
    td_idf_query=td_idf_query.reindex(sorted(a.columns), axis=1) 


    #Evaluate cosine similarity between each document and the query.


    for i,row in a.iterrows():

        num=np.dot(td_idf_query.values.flatten(),row.values)
        den=np.linalg.norm(td_idf_query.values.flatten()) * np.linalg.norm(row.values)

        cosine=num/den
        list_cosine_similarity.append(cosine)

        
        


    data["Similarity"]= list_cosine_similarity
    dd=data.sort_values("Similarity",ascending=False)
    if stars != None:
        dd=filter_dataframe_by_columns(dd, stars)
            

    
    if season==None:
        dd=dd[dd.Similarity > 0] #selezionare solo le puntate rilevanti
        #dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione',"Similarity"]] #per debug
        dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']]
    else:
        dd=dd[dd.Similarity > 0] #selezionare solo le puntate rilevanti
        #dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione',"Similarity"]][df.season==season] #per debug
        dd=dd[['season', 'episodio', 'titolo', 'trama', 'guest_star', 'prima_visione']][df.season==season]
    
    if dd.shape[0]==0: #in caso il dataframe di output è vuoto si print un messaggio di errore
        print("Nessuna puntata ritrovata: \n   - Provare ad ampliare la ricerca (es: eliminare restrizioni sulle stagioni);\n   - Controllare ciò che è stato scritto.")
        return None
    

        
    
    dd.reset_index(drop=True,inplace=True)
    #display(HTML(dd.head(k).to_html()))
    return dd
