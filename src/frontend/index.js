// Model selection dropdown

/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
function myFunction() {
    document.getElementById("modelDropdown").classList.toggle("show");
}
  
// Close the dropdown menu if the user clicks outside of it
window.onclick = function(event) {
    if (!event.target.matches('.dropbtn')) {
        var dropdowns = document.getElementsByClassName("model_selection_dropdown_content");
        var i;
        for (i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}

// Image uploading

let inpFile = document.getElementById("inputFile");
const previewContainer = document.getElementById("imagePreview");
const previewImage = previewContainer.querySelector(".image_preview_image");
const previewDefaultText = previewContainer.querySelector(".image_preview_default_text");

inpFile.addEventListener("change", function() {
    let file = this.files[0];

    if(file){
        const reader = new FileReader();

        previewDefaultText.style.display = "none";
        previewImage.style.display = "block";

        reader.addEventListener("load", function(){
            previewImage.setAttribute("src", this.result);

            let data = new FormData();
            data.append('file', file);

            const postImage = async() => {
                const response = await fetch('http://127.0.0.1:5000/testPost',{
                    method: 'POST',
                    body: data,
                    headers: {
                        'credentials': "same-origin",
                        'Origin': 'http://localhost:5500/'
                    }
                })
                .then(function(response) {
                    if (response.status !== 200)
                        console.log(`Status code: ${response.status}, error message ${response.body}`);
                    else
                        console.log(`Posted sucessfully, ${response.body}`);
                });
            }
            postImage();
            

            // const userAction = async () => {
            //     const response = await fetch('http://127.0.0.1:5000/test',{
            //         method: 'GET',
            //         headers: {
            //             'Content-Type': 'application/json',
            //             'credentials': "include",
            //             'Origin': 'http://localhost:5500/'
            //           }
            //     });
            //     console.log("waiting");
            //     const myJson = await response.json();
            //     console.log(myJson);
            // }
            // userAction();
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