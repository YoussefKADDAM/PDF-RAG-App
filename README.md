# 📘 PDF-RAG-App  
### PDF Q&A Chatbot using Ollama + LangChain + FAISS

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to:

- Upload any **PDF**
- Ask questions about its content  
- Generate **summaries** (short, detailed, bullet points)
- View the **source pages** used for each answer  
- Run everything **locally for free** using **Ollama LLaMA 3**  
- Enjoy a clean UI powered by **Streamlit**

---

## 🚀 Features

- 📄 **PDF Text Extraction** with PyPDF2  
- ✂️ **Smart Text Chunking** using LangChain  
- 🧠 **Embeddings** via OLLAMA nomic-embed-text  
- 📚 **Vector Search** with FAISS  
- 🤖 **Local LLM** (LLaMA 3) through Ollama  
- 💬 **Chat Mode** + Chat History  
- 📝 **Summary Generator** (short / detailed / bullet point)  
- 🔐 Fully local — no API keys required  

---

## 🏗️ Project Structure

│── app.py # Streamlit UI

│── rag_pipeline.py # All RAG logic (clean architecture)

│── requirements.txt # Python dependencies

│── Dockerfile # Deployment-ready container

│── .gitignore

│── .env.example

│── Images/ # Screenshots (optional)


---

## ▶️ Run Locally (Recommended)

1️⃣ Install dependencies  

pip install -r requirements.txt

2️⃣ Start Ollama

Download Ollama: https://ollama.ai

ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

3️⃣ Run the app

streamlit run app.py

App will run at:
👉 http://localhost:8502

🐳 Run with Docker

docker build -t pdf-rag-app .
docker run -p 8501:8501 pdf-rag-app

## 🛠️ Tech Stack

Python 3.11

LangChain

FAISS

Streamlit

Ollama + LLaMA 3

PyPDF2

![App Screenshot 1](Images/PNG1.png)
![App Screenshot 2](Images/PNG2.png)

## 👤 Author

Youssef Kaddam
💼 Data Science & AI
🔗 https://github.com/YoussefKADDAM
