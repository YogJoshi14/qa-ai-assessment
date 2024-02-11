---
title: Chat With Documents
emoji: 🦀
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# qp-ai-assessment
Contextual Chat Bot

Simple Contextual Chat Bot
1. Read a long PDF/ Word Document. 
2. Build a chat bot that will use the document as a context to answer the question. 
3. If the answer is not found in the document - it should say I don't know the answer. 

Advanced Challenge:
- Break down the document into multiple chunks/ paragraphs. 
- Store them in a vector database like pinecone.  
- When you ask a question find out the top 3 chunks that will likely have the answer to the question using semantic similarity search. 

#**System Design**

![Architecture](https://raw.githubusercontent.com/YogJoshi14/qp-ai-assessment/main/PDF_chat.png)

#**Required Packages**
1. Langchain : LangChain is a framework for developing applications powered by language models. [Docs](https://python.langchain.com/docs/get_started/introduction)
2. Pinecone : Pinecone makes it easy to provide long-term memory for high-performance AI applications. It’s a managed, cloud-native vector database with a simple API and no infrastructure hassles. Pinecone serves fresh, filtered query results with low latency at the scale of billions of vectors. [Docs](https://docs.pinecone.io/docs/quickstart)
3. Sentence_transformers : SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in our paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. [Docs](https://www.sbert.net/)
4. pdf2image : pdf2image is a python module that wraps the pdftoppm and pdftocairo utilities to convert PDF into images. [Docs](https://pdf2image.readthedocs.io/en/latest/overview.html)
5. pypdf2 : PyPDF2 is a free and open source pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files. It can also add custom data, viewing options, and passwords to PDF files. PyPDF2 can retrieve text and metadata from PDFs as well.[Docs](https://pdf2image.readthedocs.io/en/latest/overview.html)
6. transformers : Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. [Docs](https://huggingface.co/docs/transformers/en/index)

#**Limitations**
1. Embedding : As the project has made use of readily available huggingface embeddings, it has max dimension of 768. We can make use of alternate embeddings such as HuggingFaceInstructEmbeddings, Ollama embeddings which are open-source or OpenAI embeddings.
2. LLM : Making use of llm which has more parameter and was trained more data can also provide optimal results.

