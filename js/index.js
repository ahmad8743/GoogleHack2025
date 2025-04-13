
const button = document.querySelector('.mode-button');
let switched = true; 

button.addEventListener('click', () => {
switched = !switched;
if (switched) {
  button.textContent = "Text-to-Speech";
} else {
  button.textContent = "Speech-to-Text";
}
});