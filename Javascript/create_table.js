// Define the GitHub repository details
async function import_json(variabile_locale){

  data = sessionStorage.getItem(variabile_locale);

  var t =JSON.parse(data);

  const tkeys= Object.keys(t);
    let tableData=[];

    for (i=0; i < tkeys.length; i++ ){
      tableData.push(t[`${tkeys[i]}`]);
    }

  // Create the table element

  //Queste righe di codice servono a cancellare le vecchie tabelle, quando ne viene richiesta una nuova.
  var t=document.getElementById("tabella_output");
  var t_len =document.getElementById("tabella_output").rows.length;
  if (t_len!==0){
    t.remove()

    var table =  document.createElement('table');
    table.setAttribute("id", "tabella_output");

  }
  else{
    var table=document.getElementById("tabella_output");
  }



  // Create the table header row
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');

  // Iterate over the keys of the first object to create table headers
  Object.keys(tableData[0]).forEach(key => {
  const th = document.createElement('th');
  th.textContent = key;
  headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Create the table body
  const tbody = document.createElement('tbody');

  // Iterate over the table data and create table rows
  tableData.forEach(rowData => {
  const row = document.createElement('tr');

  // Iterate over the values of each object to create table cells
  Object.values(rowData).forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
  });

  tbody.appendChild(row);
  });

  table.appendChild(tbody);

  // Append the table to the document body or any desired element


  // Add CSS styles to make the table header sticky
  var windowHeight = window.innerHeight;

    // Calculate the desired max height (e.g., 80% of the window height)
    var maxHeight = Math.floor(windowHeight * 0.8) + 'px';
  table.style.maxHeight =  maxHeight;


  document.body.appendChild(table);
}