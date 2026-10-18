from langchain_google_genai import ChatGoogleGenerativeAI
import os


def generate_todo(context):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    prompt = f"""
Extract action items from the following text.

Return them as a numbered todo list.

Text:

{context}
"""

    return llm.invoke(prompt).content