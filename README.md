# ITC
Income tax Chatbot for 1961 and 2025 comparison
# 🧾 ITC Chatbot – Income Tax Compliance Assistant

A smart and dynamic AI-powered chatbot designed to assist users with queries related to the **Income Tax Act, 1961**. The bot provides answers by integrating both **static legal provisions** and **real-time updates** (like Finance Acts, CBDT circulars, etc.). Built using `LangChain`, `FastAPI`, and live web scraping tools, it bridges the gap between historical legislation and current financial regulations.

---

## 🚀 Features

- **Dual-Source Answering:** Combines static content from the Income Tax Act with live updates scraped from trusted government sources.
- **Context-Aware Responses:** Uses vectorstores to search for and deliver precise legal context.
- **FastAPI Backend:** Lightweight and scalable API service for interacting with the chatbot engine.
- **LangChain Integration:** Handles intelligent query chaining and document-based logic processing.
- **User-Friendly Interface (Optional):** Can be paired with a React frontend for an interactive UX.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI  
- **NLP & Logic:** LangChain, OpenAI LLMs  
- **Data Sources:** Static Legal Texts (Income Tax PDFs), Dynamic Data via Web Scraping  
- **Storage:** FAISS or Chroma Vectorstore (for embedding documents)  
- **Deployment:** Localhost / Docker / Render / Vercel (Optional UI)  
- **Optional Frontend:** React (if integrated)

---

## 📚 How It Works

1. **Document Ingestion:**  
   Loads the Income Tax Act, relevant CBDT Circulars, and budget updates into a vectorstore.

2. **Query Processing:**  
   User inputs a tax-related question. LangChain processes it and finds the most relevant documents.

3. **Dual-Response Generation:**  
   - Returns an answer based on the legal act (static).
   - Adds recent updates if applicable (via scraping or pre-indexed updates).

4. **Response Display:**  
   Clean, contextual response with citation links (if available).

---
