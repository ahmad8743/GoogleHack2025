import creds
import google.generativeai as genai
def generate():
  genai.configure(api_key=creds.API_KEY)

  model = genai.GenerativeModel(model_name="gemini-2.0-flash")

  response = model.generate_content("Generate a random sentence with 3 to 4 words.")

  return response.text
