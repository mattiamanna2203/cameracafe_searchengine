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
   
   outputdim=sessionStorage.getItem("n_output") #dimensioni massime output
   tipo_ricerca=sessionStorage.getItem("searchtype") #tipo di ricerca, keyword o ranking engine
   star=sessionStorage.getItem("star") #quali gueststar
   season=sessionStorage.getItem("season") #stagione


   if star == "None":
      star=None
   else :
      if "," in star: #Se c'è più di una guest star ci sarà una virgola e si farà lo split su di essa
         star=star.split(",")
      else:       #se non c'è la virgola allora basta tenersi la stringa e metterla in una lista
         star=[star]

   if season == "None":
      season=None
   else:
      if "," in season: #Se c'è più di una stagione ci sarà una virgola e si farà lo split su di essa
         season=season.split(",")
         season=list(map(int,season)) #trasformare gli elementi della lista in numeri
      else:
         season=[int(season)]  #se non c'è la virgola allora basta tenersi la stringa e metterla in una lista, prima però si trasforma la stringa in un intero

   if "Più attinenti" ==tipo_ricerca:
      if outputdim == "Tutti":
         output = query_ranking(stringa.value,None,season,star) 

      else:
         #Richiamare la funzione
         output = query_ranking(stringa.value,int(outputdim),season,star) 

   elif "Tutte le keyword"==tipo_ricerca:
      if outputdim == "Tutti":
         output=query_ranking_allMatch(stringa.value,None,season) 
      else:
         output=query_ranking_allMatch(stringa.value,int(outputdim),season) 


   if type(output)== type(None):
      a=[{"season":'',"episodio":"","titolo":"",
"trama":"","guest_star":'',"prima_visione":""}]
      d=pd.DataFrame.from_dict(a)
      sessionStorage.setItem('database',d.to_json(orient='records')) 
      js.import_json() 
   else:
      sessionStorage.setItem('database',output.to_json(orient='records'))  

      #Scrivendo js.nomedellafunzione, si potrà utilizzare    una funzione presente nei file JavaScript importati. 
      #Permette quindi di far lavorare linguaggio Python3 e JavaScript insieme. 
      js.import_json() 

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

#%% Gestire filters
def season_call(*arg):
   outputdim=sessionStorage.getItem("n_output")
   season=sessionStorage.getItem("season") #stagione


   if season == "None":
      season = None
   else:
      if "," in season: #Se c'è più di una stagione ci sarà una virgola e si farà lo split su di essa
         season=season.split(",")
         season=list(map(int,season)) #trasformare gli elementi della lista in numeri
      else:
         season=[int(season)]  #se non c'è la virgola allora basta tenersi la stringa e metterla in una lista, prima però si trasforma la stringa in un intero

   if outputdim == "Tutti":
      outputdim = None
   else:
      outputdim = int(outputdim)

   output=only_season(season,outputdim)
   if type(output)== type(None):
      a=[{"season":'',"episodio":"","titolo":"",
"trama":"","guest_star":'',"prima_visione":""}]
      d=pd.DataFrame.from_dict(a)
      sessionStorage.setItem('database',d.to_json(orient='records')) 
      js.import_json() 
   else:
      sessionStorage.setItem('database',output.to_json(orient='records'))  

      #Scrivendo js.nomedellafunzione, si potrà utilizzare    una funzione presente nei file JavaScript importati. 
      #Permette quindi di far lavorare linguaggio Python3 e JavaScript insieme. 
      js.import_json() 

#Selezionare il tasto di avvio ricerca
season_button = document.getElementById("only-season") 

#Aggiungere un event listener al tasto di avvio ricerca, quando si premerà il tasto verrà richiamata la funzione avvio_ricerca(*arg)
add_event_listener(season_button, "click", season_call)



def starcall(*arg):
   star=sessionStorage.getItem("star")
   season=sessionStorage.getItem("season") #stagione
   if star == "None":
      star=None
   else :
      if "," in star: #Se c'è più di una guest star ci sarà una virgola e si farà lo split su di essa
         star=star.split(",")
      else:       #se non c'è la virgola allora basta tenersi la stringa e metterla in una lista
         star=[star]
   if season == "None":
      season = None
   else:
      if "," in season: #Se c'è più di una stagione ci sarà una virgola e si farà lo split su di essa
         season=season.split(",")
         season=list(map(int,season)) #trasformare gli elementi della lista in numeri
      else:
         season=[int(season)]  #se non c'è la virgola allora basta tenersi la stringa e metterla in una lista, prima però si trasforma la stringa in un intero
   output=only_star(star,season)
   if type(output)== type(None):
      a=[{"season":'',"episodio":"","titolo":"",
      "trama":"","guest_star":'',"prima_visione":""}]
      d=pd.DataFrame.from_dict(a)
      sessionStorage.setItem('database',d.to_json(orient='records')) 
      js.import_json() 
   else:
      sessionStorage.setItem('database',output.to_json(orient='records'))  

      #Scrivendo js.nomedellafunzione, si potrà utilizzare    una funzione presente nei file JavaScript importati. 
      #Permette quindi di far lavorare linguaggio Python3 e JavaScript insieme. 
      js.import_json() 
#Selezionare il tasto di avvio ricerca
star_button = document.getElementById("only-guest-star") 

#Aggiungere un event listener al tasto di avvio ricerca, quando si premerà il tasto verrà richiamata la funzione avvio_ricerca(*arg)
add_event_listener(star_button, "click", starcall)

