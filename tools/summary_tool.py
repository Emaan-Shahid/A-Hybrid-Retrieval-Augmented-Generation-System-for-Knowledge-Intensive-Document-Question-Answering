from langchain_google_genai import ChatGoogleGenerativeAI
import os


def summarize_document(context):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    prompt = f"""
You are an expert summarizer.

Summarize the following document.

Use bullet points.

Keep it concise.

Document:

{context}
"""

    return llm.invoke(prompt).content