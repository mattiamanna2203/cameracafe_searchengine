// Get references to the button and the modal
var showModalButton = document.getElementById("filter");
var modal = document.getElementById("myModal");

// Show the modal when the button is clicked
showModalButton.addEventListener("click", function(){
    let stato_filtri = sessionStorage.getItem("filtri");
    if ( stato_filtri == "close"){
        modal.style.display = "block";
        sessionStorage.setItem("filtri","open");
    }
    else if  ( stato_filtri == "open"){
        modal.style.display = "none";
        sessionStorage.setItem("filtri","close");
    }
});

// Close the modal when clicking anywhere outside of it
window.addEventListener("click", function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
        sessionStorage.setItem("filtri","close");
    }
});

const closeModalButton = document.getElementById("closeModal");
closeModalButton.addEventListener("click", function() {
    modal.style.display = "none";
    sessionStorage.setItem("filtri","close");
});
