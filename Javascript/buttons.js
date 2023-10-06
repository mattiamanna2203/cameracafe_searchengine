const button= document.querySelector(".avvio");
const stringa= document.querySelector(".barra-ricerca");
button.addEventListener('click',() =>{console.log('ciao');   console.log(stringa.value);}) 



// Construct the URL to the raw content of the file
const rawFileURL = "https://raw.githubusercontent.com/mattiamanna2203/cameracafe_searchengine/master/Dati/light_vocabulary.json";
