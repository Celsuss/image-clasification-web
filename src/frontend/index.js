// Global elements
const imageUploadContainer = document.getElementById("imageUploadAndPreview");
const resultTable = document.getElementById('resultTable');
// Global variables
var model;
var data;

function updateModelPickedList(){
    model_pick_elements = document.getElementsByClassName("model_pick");
    var dropdown_btn_element = document.getElementById("modelDropdownBtn");
    for(var i = 0; i < model_pick_elements.length; i++){
        var model_pick_element = model_pick_elements[i];
        model_pick_element.addEventListener("click",  function() {
            var value = this.text;
            console.log("Model picked " + value);
            model = value;
            dropdown_btn_element.innerHTML = value;
            imageUploadContainer.style.display = "block";

            document.getElementById("resultText").style.display = "none";
            resultTable.style.display = "none";
        });
    }
}

// Populate model selection list
function populateModelList(model_list){
    var dropdown_element = document.getElementById("modelDropdown");

    for (var i = 0; i < model_list["models"].length; i++){
        var para = document.createElement("a");
        para.className = "model_pick";
        var node = document.createTextNode(model_list["models"][i]);
        para.appendChild(node);

        dropdown_element.appendChild(para);
    }

    updateModelPickedList();
}

// Get model list
const fetchModelList = async () => {
    const response = await fetch('http://127.0.0.1:5000/list_model',{
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'credentials': "include",
            'Origin': 'http://localhost:5500/'
            }
    })
    .then((response) => response.json())
    .then((responseData) => {
          console.log(responseData);
          populateModelList(responseData);
        })
        .catch(error => console.warn(error));
}

fetchModelList();

/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
function modelDropdown() {
    document.getElementById("modelDropdown").classList.toggle("show");
}
  
// Close the dropdown menu if the user clicks outside of it
window.onclick = function(event) {
    if (!event.target.matches('.dropbtn')) {
        var dropdowns = document.getElementsByClassName("model_selection_dropdown_content");
        for (var i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}

// Image uploading
const inpFile = document.getElementById("inputFile");
const previewContainer = document.getElementById("imagePreview");
const previewImage = previewContainer.querySelector(".image_preview_image");
const previewDefaultText = previewContainer.querySelector(".image_preview_default_text");

function uploadImage() {
    const postImage = async() => {
        const response = await fetch('http://127.0.0.1:5000/post_image',{
            method: 'POST',
            body : data,
            headers: {
                'credentials': "same-origin",
                'Origin': 'http://localhost:5500/'
            }
        })
        .then(function(response) {
            if (response.status !== 200){
                response.json().then(function(body){
                    console.log(`Status code: ${response.status}, error message ${body["msg"]}`);
                    document.getElementById("resultText").innerHTML = body["msg"];
                });
            }
            else{
                console.log("Got response");
                response.json().then(function(body){
                    console.log(body);
                    document.getElementById("resultText").innerHTML = 'Results:';
                    tableCreate(body['label'], body['score']);
                    resultTable.style.display = "block";
                });
            }
        });
    }
    postImage();

    document.getElementById("resultText").style.display = "block"; 

    function tableCreate(label, score){
        // clean up the old table
        while(resultTable.hasChildNodes())
        {
            resultTable.removeChild(resultTable.firstChild);
        }

        var columns = ["#", "Label", "Score"];
        // create new table 
        for(var i = 0; i < label.length+1; i++){
            var tr = resultTable.insertRow();
            for(var j = 0; j < 3; j++){
                if(i == 0){
                    var td = tr.insertCell();
                    td.appendChild(document.createTextNode(columns[j]));
                } else {
                var td = tr.insertCell();
                if (j == 0){
                    td.appendChild(document.createTextNode(i));
                }
                else if(j == 1){
                    td.appendChild(document.createTextNode(label[i-1]));
                }else{
                    td.appendChild(document.createTextNode(score[i-1].substring(0,5)));
                }
            }
            td.style.border = '1px solid black';
            }
        }
    }
}

inpFile.addEventListener("change", function() {
    const file = this.files[0];

    if(file){
        document.getElementById("resultText").style.display = "none";
        document.getElementById("resultText").innerHTML = `Running prediction ...`;
        resultTable.style.display = "none";

        console.log("Using model: " + model);
        const reader = new FileReader();

        previewDefaultText.style.display = "none";
        previewImage.style.display = "block";

        reader.addEventListener("load", function(){
            previewImage.setAttribute("src", this.result);

            data = new FormData();
            data.append('model_name', model);
            data.append('file', file);
        });

        reader.readAsDataURL(file);
    }
    else{
        previewDefaultText.style.display = null;
        previewImage.style.display = null;
        previewImage.setAttribute("src", "");
    }

    console.log(file)
});