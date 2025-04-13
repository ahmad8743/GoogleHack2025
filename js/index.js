
const button = document.querySelector('.mode-button');
const input = document.querySelector('.input');
const output = document.querySelector('.output');

let switched = true; 

button.addEventListener('click', () => {
  switched = !switched;
  if (switched) {
    button.textContent = "Text-to-Speech";
    output.classList.remove('invisible');

    input.classList.add('invisible');
  } else {
    button.textContent = "Speech-to-Text";
    input.classList.remove('invisible');
    output.classList.add('invisible');
   
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
