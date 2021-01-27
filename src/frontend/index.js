// Image uploading

const inpFile = document.getElementById("inputFile")
const previewContainer = document.getElementById("imagePreview")
const previewImage = previewContainer.querySelector(".image_preview_image")
const previewDefaultText = previewContainer.querySelector(".image_preview_default_text")

inpFile.addEventListener("change", function() {
    const file = this.files[0];

    if(file){
        const reader = new FileReader();

        previewDefaultText.style.display = "none";
        previewImage.style.display = "block";

        reader.addEventListener("load", function(){
            previewImage.setAttribute("src", this.result);

            const userAction = async () => {
                const response = await fetch('http://127.0.0.1:5000/test',{
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'credentials': "include",
                        'Origin': 'http://localhost:5500/'
                      }
                });
                console.log("waiting");
                const myJson = await response.json();
                console.log(myJson);
            }
            userAction();
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