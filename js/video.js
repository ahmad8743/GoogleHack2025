// Get references to DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const gestureOutput = document.getElementById('gesture');
let model;

// Set up the camera using getUserMedia
export async function setupCamera() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }});
    video.srcObject = stream;
    return new Promise(resolve => {
      video.onloadedmetadata = () => { resolve(video); };
    });
  } else {
    alert("Your browser does not support access to the camera.");
  }
}

// Load the Handpose model
export async function loadModel() {
  model = await handpose.load();
  console.log("Handpose model loaded.");
}

// Process each video frame
export async function detectHands() {
  const predictions = await model.estimateHands(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (predictions.length > 0) {
    predictions.forEach(prediction => {
      // Draw landmarks for each detected hand
      const landmarks = prediction.landmarks;
      landmarks.forEach(point => {
        const [x, y] = point;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });
    });

    // Placeholder logic: If at least one hand is detected, output "A"
    // In a real application, you'd pass "predictions" to your gesture classifier here.
    gestureOutput.innerText = "A (dummy result)";
  } else {
    gestureOutput.innerText = "No hand detected";
  }
  requestAnimationFrame(detectHands);
}