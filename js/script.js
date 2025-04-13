import "./video.js";
import "./index.js";

async function main() {
  await setupCamera();
  video.play();
  await loadModel();
  detectHands();
}

main();