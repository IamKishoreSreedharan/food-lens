async function analyzeImage() {
    let input = document.getElementById('imageUpload');
    let formData = new FormData();
    formData.append("image", input.files[0]);

    // Make a POST request to the back-end FastAPI server
    let response = await fetch('http://localhost:5000/analyze/', {
        method: 'POST',
        body: formData
    });

    let result = await response.json();
    document.getElementById('result').innerText = "Prediction: " + result.prediction;
}
