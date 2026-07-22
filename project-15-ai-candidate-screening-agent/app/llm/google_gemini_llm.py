from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

class GoogleGeminiLLM:
    def __init__(self):
        load_dotenv()
        print("API KEY:", os.getenv("GOOGLE_API_KEY"))

    def get_llm(self):
        try:
             # Gemini for answer generation only
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY"))
            return llm
        except Exception as e:
            raise ValueError(f"Error occured with exception: {e}")