import streamlit as st
from transformers import pipeline
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# ----------------------
# 1. Load Models
# ----------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Use a better SpaCy model if available (sm = small, trf = transformer)
nlp = spacy.load("en_core_web_sm")

# ----------------------
# 2. Summarization Function
# ----------------------
def generate_summary(text):
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# ----------------------
# 3. Knowledge Graph Builder with Filters
# ----------------------
def build_knowledge_graph(text):
    doc = nlp(text)
    G = nx.Graph()

    for ent in doc.ents:
        # Only keep meaningful entities (People, Organizations, Locations)
        if ent.label_ in ["PERSON", "ORG", "GPE"] and len(ent.text.strip()) > 2:
            G.add_node(ent.text, label=ent.label_)

    # Connect entities appearing in the same sentence
    for sent in doc.sents:
        ents = [ent.text for ent in sent.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                G.add_edge(ents[i], ents[j])

    return G

# ----------------------
# 4. Graph Visualization
# ----------------------
def visualize_graph(G):
    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(G)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    return tmp_file.name

# ----------------------
# 5. Streamlit UI
# ----------------------
st.title("📝 Generative AI Summarizer with Knowledge Graph")
st.write("Enter an article or long text. The app will generate a summary and build a knowledge graph of entities.")

user_input = st.text_area("Paste your text here:", height=250)

if st.button("Generate Summary & Graph"):
    if user_input.strip():
        st.subheader("📌 Summary")
        summary = generate_summary(user_input)
        st.success(summary)

        st.subheader("🌐 Knowledge Graph")
        G = build_knowledge_graph(user_input)

        if len(G.nodes) == 0:
            st.warning("⚠️ No valid entities found (PERSON, ORG, GPE). Try different text.")
        else:
            graph_path = visualize_graph(G)
            HtmlFile = open(graph_path, 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=550)
    else:
        st.warning("⚠️ Please enter some text to summarize.")