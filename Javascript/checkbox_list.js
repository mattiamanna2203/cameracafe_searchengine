//https://blog.hubspot.com/website/html-dropdown
$(function()
{
   var mySelectCheckbox = new checkbox_select(
   {
      selector : "#make_checkbox_select_gueststar",
         selected_translation : "Selezionati",
         all_translation : "Totale selezionati: ",
         not_found : "Non trovato",

      // Event during initialization
      onApply : function(e)
      {
         console.log("Conferma");
         console.log(e.selected);
      }
   });

});


$(function()
{
   var mySelectCheckbox = new checkbox_select(
   {
      selector : "#make_checkbox_select",
         selected_translation : "selezionati",
         all_translation : "Totale selezionati: ",
         not_found : "Non trovato",

      // Event during initialization
      onApply : function(e)
      {
         console.log("Conferma");
         console.log(e.selected);
      }
   });

});




$(function()
{
   var mySelectCheckbox = new radio_select(
   {
      selector : "#make_checkbox_select_numberoutput",
         selected_translation : "selezionati",
         all_translation : "Totale selezionati: ",
         not_found : "Non trovato",

      // Event during initialization
      onApply : function(e)
      {
         console.log("Conferma");
         console.log(e.selected);
      }
   });

});


$(function()
{
   var mySelectCheckbox = new radio_select(
   {
      selector : "#make_checkbox_select_searchtype",
         selected_translation : "selezionate",
         all_translation : "Totale selezionati: ",
         not_found : "Non trovato",

      // Event during initialization
      onApply : function(e)
      {
         console.log("Conferma");
         console.log(e.selected);
      }
   });

});





