import requests
import base64
import creds

url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={creds.API_KEY}"

inputText = "This is a test" # I expect the text from the translated ASL (english) to be here

payload = {
    "input": {"text": "What the hellyyonte"},
    "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-D"},
    "audioConfig": {
        "audioEncoding": "MP3",
        "speakingRate": 1.0
        }
}

response = requests.post(url, json=payload) 

# The code below produces an mp3 file in the directory which should be played iff user presses a text to speech button
if response.ok:
    audio_content = response.json()['audioContent']
    with open('output.mp3', 'wb') as audio_file:
        audio_file.write(base64.b64decode(audio_content))
    print("Audio file saved as output.mp3")
else:
    print(f"Request failed: {response.status_code}, {response.text}")
