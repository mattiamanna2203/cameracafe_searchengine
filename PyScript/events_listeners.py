#%% Importazione pacchetti
from js import console, document, sessionStorage
   #Scrivendo from js ... si importano direttamente le funzioni e funzionalità  JavaScript indicate
      # - console,        permette di utilizzare i console.log();
      # - sessionStorage, permette di utilizzare le variabili locali;
      # - document,       permette di selezionare elementi dell'html;
from pyodide.ffi.wrappers import add_event_listener
#Permette di utilizzare gli event listener, tuttavia sono leggermente diversi da quelli JavaScript, per utilizzarli correttamente andrà usata una funzione in più.


#%% Funzione che avvia la ricerca.
def avvio_ricerca(*arg):
   """
      Funzione che associa all'event listener per al tasto di avvio di ricerca("#avvio-ricerca") la pipeline python da seguire.
      Lavorando con PyScript questo è l'unico modo che  per ora ho trovato per utilizzare correttamente degli eventlistener.
   """
  stringa= document.querySelector(".barra-ricerca") #Estrarre la stringa digitata dall'utente.
  output = query_ranking(stringa.value,100) #Richiamare la funzione
  sessionStorage.setItem('database',output.to_json(orient='records')) #Salvare localmente un file.
  dati = sessionStorage.getItem('database'); # Utilizzare un file salvato localmente.
  js.import_json('database') #Scrivendo js.nomedellafunzione, si potrà utilizzare    una funzione presente nei file JavaScript importati. 
                             #Permette quindi di far lavorare linguaggio Python3 e JavaScript insieme.

avv = document.getElementById("avvio-ricerca") #Puntare
add_event_listener(avv, "click", avvio_ricerca)