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
                const response = await fetch('localhost', {
                method: 'POST',
                body: this.result, // string or object
                headers: {
                    'Content-Type': 'application/json'
                }
                });
            }
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

// Rest API communication
fetch('/')
      .then(function (response) {
          return response.json();
      }).then(function (text) {
          console.log('GET response:');
          console.log(text.greeting); 
      });