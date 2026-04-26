const predictButton = document.getElementById("predict-btn");
const predictionText = document.getElementById("prediction-text");

async function requestPrediction() {
  predictionText.textContent = "Predicting...";

  try {
    const response = await fetch("/predict");
    const data = await response.json();

    if (!response.ok) {
      predictionText.textContent = data.error || "Prediction failed.";
      return;
    }

    predictionText.textContent = `Word: ${data.prediction} (frames used: ${data.frames_used})`;
  } catch (error) {
    predictionText.textContent = "Network error while requesting prediction.";
  }
}

predictButton.addEventListener("click", requestPrediction);
