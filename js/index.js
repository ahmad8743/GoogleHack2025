const button = document.querySelector('.mode-button');
const input = document.querySelector('.input');
const output = document.querySelector('.output');
const play = document.querySelector('.play-button');

let switched = true; 

button.addEventListener('click', () => {
  event.preventDefault();
  switched = !switched;
  if (switched) {
    button.textContent = "Text-to-Speech";
    output.classList.remove('invisible');
    play.classList.remove('invisible');
    input.classList.add('invisible');
  } else {
    button.textContent = "Quiz Mode";
    input.classList.remove('invisible');
    output.classList.add('invisible');
    play.classList.add('invisible');
   
  }
});

setInterval(function() {
    fetch('http://127.0.0.1:5001/get_prediction')
        .then(response => response.json())
        .then(data => {
            console.clear()
            console.log("Latest prediction:", data.prediction);
            // Update your webpage with the new prediction here.
        })
        .catch(err => console.error('Error fetching prediction', err));
}, 2000);  // Polling every 2000 milliseconds (2 seconds)



const sentence = document.querySelector('.sentence-text');
sentence.innerText = generate();
async function updateSentence() {
  try {
    // Call the Flask API endpoint
    const response = await fetch('http://127.0.0.1:5000/generate');
    const data = await response.json();
    // Update the HTML element with the generated sentence
    sentence.textContent = data.text;
  } catch (error) {
    console.error("Error fetching generated text:", error);
  }
}

// Update the sentence when the page loads
document.addEventListener("DOMContentLoaded", updateSentence);