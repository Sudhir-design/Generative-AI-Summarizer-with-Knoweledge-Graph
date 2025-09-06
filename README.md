#  Generative AI Summarizer with Knowledge Graph

This project is a **Streamlit web application** that uses **Hugging Face Transformers (BART)** to summarize long text and **SpaCy + NetworkX** to extract named entities and build an interactive knowledge graph.

---

##  Features
-  **Summarization** → Generates concise summaries of long text using Hugging Face `facebook/bart-large-cnn`  
-  **Knowledge Graph** → Extracts people, organizations, and locations with SpaCy (NER) and builds an interactive graph using PyVis  
-  **Streamlit UI** → Clean web interface to paste text and visualize results  
-  **Entity Filtering** → Keeps only meaningful entities (PERSON, ORG, GPE) for clarity  
-  **Future Work** → Add automatic ROUGE evaluation for summaries  

---

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (web app)  
- **Hugging Face Transformers** (summarization)  
- **SpaCy** (named entity recognition)  
- **NetworkX + PyVis** (graph visualization)  

---

## 📂 Project Structure
