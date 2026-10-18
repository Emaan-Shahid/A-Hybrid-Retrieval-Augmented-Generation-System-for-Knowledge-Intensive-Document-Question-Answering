from langchain_google_genai import ChatGoogleGenerativeAI
import os


def generate_email(context):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    prompt = f"""
Write a professional email based on the following information.

Information:

{context}
"""

    return llm.invoke(prompt).content