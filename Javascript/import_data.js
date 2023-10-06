async function import_json(filename){
   const path = `${filename}`;

  const url = `https://raw.githubusercontent.com/mattiamanna2203/cameracafe_searchengine/master/Dati/${path}.json`;
   const response = await fetch(url);

   console.log(url);
   const oggetto_response= await response.json();


/*https://raw.githubusercontent.com/mattiamanna2203/cameracafe_searchengine/master/Dati/light_word_dict.json
*/
   const data_url=oggetto_response.download_url;
   // Fetch the CSV file from the download URL
   const data_raw = await fetch(data_url);

   const data = await data_raw.text();

   console.log(data);
   var t =JSON.parse(data);
   console.log(t);
}
//import_json("light_word_dict")
