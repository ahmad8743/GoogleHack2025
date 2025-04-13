
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


