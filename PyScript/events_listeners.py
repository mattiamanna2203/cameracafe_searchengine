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
   Funzione che associa all'event listener del tasto di avvio di ricerca("#avvio-ricerca") la pipeline python da seguire.

   Lavorando con PyScript questo è l'unico modo che  per ora ho trovato per utilizzare correttamente degli eventlistener.

   Questa funzione non necessita argomenti, le informazioni necessarie sono direttamente ricavate dall'html tramite querySelector.
   """
   #Estrarre la stringa digitata dall'utente.
   stringa = document.querySelector(".barra-ricerca")

   #Richiamare la funzione
   output = query_ranking(stringa.value,100) 

   #Salvare localmente un file.
   sessionStorage.setItem('database',output.to_json(orient='records')) 

   # Utilizzare un file salvato localmente.
   dati = sessionStorage.getItem('database'); 

   #Scrivendo js.nomedellafunzione, si potrà utilizzare    una funzione presente nei file JavaScript importati. 
   #Permette quindi di far lavorare linguaggio Python3 e JavaScript insieme. 
   js.import_json('database') 


#Aggiungere un event listener alla pressione di un tasto
def handle_keypress(event):
   """
   Funzione che associa all'event listener PRESSIONE TASTO ENTER la pipeline python da seguire.

   Questa funzione non necessita argomenti, le informazioni necessarie sono direttamente ricavate dall'html tramite querySelector.

   Questa funzione permette di elaborare il tasto cliccato dall'utente. Così da assegnare al tasto una funzione specifica. 

   Informazioni trovate al link: https://www.cnblogs.com/xgqfrms/p/16513674.html
   """
   if event.key =="Enter":
      avvio_ricerca()



#Selezionare il tasto di avvio ricerca
avv = document.getElementById("avvio-ricerca") 

#Aggiungere un event listener al tasto di avvio ricerca, quando si premerà il tasto verrà richiamata la funzione avvio_ricerca(*arg)
add_event_listener(avv, "click", avvio_ricerca)

#Aggiungere un event listener alla pressione del tasto ENTER. Quando si premerà il tasto verrà richiamata la funzione  avvio_ricerca(*arg)  passando per la funzione handle_keypress che verifica che il tasto premuto sia Enter.
add_event_listener(document.body,"keypress",handle_keypress)