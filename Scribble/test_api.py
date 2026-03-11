import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API key loaded: {bool(api_key)}")
if not api_key:
    print("No API key found")
else:
    try:
        print("Configuring genai...")
        genai.configure(api_key=api_key)
        print("Creating model...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Generating content...")
        response = model.generate_content("Test message")
        print("Success:", response.text)
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()