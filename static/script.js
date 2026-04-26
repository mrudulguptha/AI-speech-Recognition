const videoElement = document.getElementById("video");
const subtitleElement = document.getElementById("subtitle");

// Reuse one canvas context to avoid repeated DOM allocations.
const canvas = document.createElement("canvas");
canvas.width = 224;
canvas.height = 224;
const canvasContext = canvas.getContext("2d");

let isPredicting = false;
let frameBuffer = [];
let predictionHistory = [];
let lastStablePrediction = "";
let lastSubtitleText = "";

function updateSubtitle(text) {
  // Skip redundant DOM writes to avoid visual jitter.
  if (lastSubtitleText !== text) {
    subtitleElement.innerText = text;
    lastSubtitleText = text;
  }
}

function addPredictionToHistory(prediction) {
  predictionHistory.push(prediction);

  // Keep only the latest 5 predictions for smoothing.
  if (predictionHistory.length > 5) {
    predictionHistory.shift();
  }
}

function getStablePrediction() {
  if (predictionHistory.length === 0) {
    return "";
  }

  const counts = {};
  for (const word of predictionHistory) {
    counts[word] = (counts[word] || 0) + 1;
  }

  let stablePrediction = predictionHistory[predictionHistory.length - 1];
  let maxCount = 0;

  for (const [word, count] of Object.entries(counts)) {
    if (count > maxCount) {
      maxCount = count;
      stablePrediction = word;
    }
  }

  return stablePrediction;
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoElement.srcObject = stream;
    updateSubtitle("Listening...");
  } catch (error) {
    updateSubtitle("Camera access denied or unavailable.");
  }
}

function captureFrameBase64() {
  // Capture and resize to 224x224 to match backend preprocessing.
  canvasContext.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg");
}

function captureFrames() {
  if (!videoElement.srcObject) {
    updateSubtitle("Listening...");
    return;
  }

  if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
    return;
  }

  const frameBase64 = captureFrameBase64();
  frameBuffer.push(frameBase64);

  // Keep only the latest 20 frames for sequence prediction.
  if (frameBuffer.length > 20) {
    frameBuffer.shift();
  }

  if (!isPredicting && frameBuffer.length < 20) {
    updateSubtitle("Listening...");
  }
}

async function sendSequenceForPrediction() {
  if (isPredicting) {
    return;
  }

  if (frameBuffer.length < 20) {
    updateSubtitle("Listening...");
    return;
  }

  isPredicting = true;
  updateSubtitle("Processing...");

  // Send a snapshot copy so ongoing capture does not mutate in-flight payload.
  const sequencePayload = frameBuffer.slice(-20);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ frames: sequencePayload }),
    });

    const data = await response.json();

    if (!response.ok) {
      updateSubtitle(data.error || "Prediction failed.");
      return;
    }

    const prediction = (data.prediction || "").trim();

    if (!prediction || prediction === "Collecting frames...") {
      updateSubtitle("Listening...");
      return;
    }

    addPredictionToHistory(prediction);
    const stablePrediction = getStablePrediction();

    // Update only when the stable word changes meaningfully.
    if (stablePrediction && stablePrediction !== lastStablePrediction) {
      lastStablePrediction = stablePrediction;
      updateSubtitle(stablePrediction);
    } else if (!lastStablePrediction) {
      updateSubtitle("Listening...");
    } else {
      // Keep the previously rendered stable subtitle and avoid flicker.
      updateSubtitle(lastStablePrediction);
    }
  } catch (error) {
    updateSubtitle("Listening...");
  } finally {
    isPredicting = false;
  }
}

startCamera();
setInterval(captureFrames, 100);
setInterval(sendSequenceForPrediction, 2000);
