# Week 2 GenAI Project: Local PDF Q&A App

This is my final project for Week 2 of the MSOC Generative AI course. I built a RAG (Retrieval-Augmented Generation) app from scratch. Basically, you upload a PDF, and you can chat with it. 

The main goal of this project was to stop the AI from making things up (hallucinating). If the answer isn't actually written in the uploaded PDF, the app is strictly programmed to just admit it doesn't know.

## A look at the app
<img width="1918" height="1068" alt="Screenshot 2026-06-14 005551" src="https://github.com/user-attachments/assets/3d706610-4d36-4f3a-9ab1-d94637bf43dd" />


## Proving it doesn't hallucinate
I tested the app to make sure it actually follows the rules:
1. **When I ask a real question:** It finds the right paragraph, gives me the answer, and even cites the exact page number it got the info from.
2. **When I ask a trick question (like "Who teaches computer networks?"):** It completely refuses to answer and tells me the info isn't in the document.

<img width="1918" height="1078" alt="Screenshot 2026-06-14 005637" src="https://github.com/user-attachments/assets/9a4ce208-c374-4756-b305-ad04eeb5505d" />


## How it works (in plain English)

If you're wondering how the code actually reads the files, here are the two main tricks I used:

### 1. Chopping the text up (Chunking)
You can't feed a massive 100-page PDF into an AI all at once because its memory gets overwhelmed. To fix this, my code chops the PDF into small, bite-sized paragraphs (about 500 characters each). I also made the code overlap these chunks a little bit. That way, if a sentence is right on the edge of a cut, it doesn't accidentally get sliced in half and lose its meaning.

### 2. Turning words into math (Embeddings)
Computers don't really understand English words, but they are amazing at math. "Embeddings" is just a fancy term for translating a sentence into a long list of numbers. 

When you upload a PDF, the app translates all those text chunks into numbers and saves them to a local database folder (using a tool called ChromaDB). Later, when you ask a question in the chat, the app turns your question into numbers too. It quickly searches the database to find the paragraphs that are mathematically closest to your question, pulls out that exact text, and hands it to the AI to answer you.

## How to run it on your machine
1. Make sure you have your Groq API key saved in a `.env` file like this: 
   `GROQ_API_KEY1=your_key_here`
2. Install the required tools by running:
   ```bash
   pip install -r requirements.txt

   ## ⚙️ Tech Stack
* **UI:** Gradio
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **LLM Provider:** Groq (`llama-3.3-70b-versatile`)
* **Framework:** LangChain
