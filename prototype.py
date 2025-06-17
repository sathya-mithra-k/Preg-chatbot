import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("dummy.json", "r") as f:
    data = json.load(f)

model = genai.GenerativeModel("gemini-pro")

def search(Uinput):
    input1 = Uinput.lower()
    for key,value in data.items():
        if any(word in input1 for word in key.split("_")):
            return value 
    return None

def response(Uinput):
    prompt = f"You are a helpful assistant for Indian mothers. Respond in a simple, respectful tone. Query:\n{Uinput}"
    response = model.generate_content(prompt)
    return response.text.strip()

def chat():
    print("Maasi: Aske me your doubts")
    Uinput = input() 
    answer = search(Uinput)
    if answer:
        print("Maasi: ", answer)
    else:
        print(" Maasi (Gemini):", response(Uinput))

if __name__ == "__main__":
    chat()