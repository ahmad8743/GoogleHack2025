import { API_KEY} from "./creds.js";
const button = document.querySelector('.mode-button');
const input = document.querySelector('.input');
const output = document.querySelector('.output');
const tts_submit_button = document.querySelector('#play-button');
const url = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${API_KEY}`;

let switched = true;
let sign_to_speech = true;
let tts_string = ""

button.addEventListener('click', () => {
  switched = !switched;
  if (switched) {
    button.textContent = "Text-to-Speech";
    output.classList.remove('invisible');
    sign_to_speech = true;

    input.classList.add('invisible');
  } else {
    button.textContent = "Speech-to-Text";
    input.classList.remove('invisible');
    output.classList.add('invisible');
    sign_to_speech = false

  }
});

document.addEventListener('keydown', (event) => {
    if (event.key === "Backspace" && sign_to_speech) {
        tts_string = tts_string.substring(0, tts_string.length - 1);
        document.getElementById("gesture").innerHTML = tts_string;
    }
});

tts_submit_button.addEventListener('click', () => {
    const requestData = {
        "input": {"text": tts_string},
        "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-D"},
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 1.0
            }
    };

  fetch(url, {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
          //'Authorization': `Bearer ${API_KEY}`  // If the API requires an Authorization header
      },
      body: JSON.stringify(requestData)
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok: ' + response.statusText);
      }
      return response.json();
  })
  .then(data => {
      console.log("Response from Gemini TTS API:", data);

      // Assume data.audioContent is a base64 encoded MP3 string.
      const base64Audio = data.audioContent;
      if (base64Audio) {
          // Create an Audio element with a data URL.
          const audioElement = new Audio("data:audio/mp3;base64," + base64Audio);
          audioElement.play().then(() => {
              console.log("Playback started successfully.");
          }).catch(error => {
              console.error("Error during playback:", error);
          });
      } else {
          console.error("No audio content received from Gemini TTS API.");
      }
  })
  .catch(error => {
      console.error("Error calling Gemini TTS API:", error);
  });

  tts_string = "";
  document.getElementById("gesture").innerHTML = tts_string;

});



setInterval(function() {
    fetch('http://127.0.0.1:5001/get_prediction')
        .then(response => response.json())
        .then(data => {
            console.log("Latest prediction:", data.prediction);
            if (sign_to_speech){
                tts_string += data.prediction;
                document.getElementById("gesture").innerHTML = tts_string;
            }
            // Update your webpage with the new prediction here.
        })
        .catch(err => console.error('Error fetching prediction', err));
}, 2000);  // Polling every 2000 milliseconds (2 seconds)
