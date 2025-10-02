{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNe8eAd7kFBVIc03Tl+O9qM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Hassan7196/Rag-StreamLit/blob/main/rag_streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vtExN7DZBfuq",
        "outputId": "e3770018-ea90-4117-ea63-ffd59c345ca5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-10-02 18:09:25.011 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.014 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.017 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.021 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.023 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.026 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.028 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.032 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-10-02 18:09:25.035 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "from langchain_community.document_loaders import PyPDFLoader\n",
        "from langchain.text_splitter import CharacterTextSplitter\n",
        "from langchain_community.embeddings import HuggingFaceEmbeddings\n",
        "from langchain_community.vectorstores import FAISS\n",
        "from langchain.chains import RetrievalQA\n",
        "from langchain_community.llms import HuggingFaceHub\n",
        "\n",
        "# --- Streamlit UI ---\n",
        "st.title(\"📑 RAG Demo - Ask Questions about Your PDF\")\n",
        "\n",
        "# File uploader\n",
        "uploaded_file = st.file_uploader(\"Upload a PDF\", type=[\"pdf\"])\n",
        "\n",
        "if uploaded_file:\n",
        "    with open(\"uploaded.pdf\", \"wb\") as f:\n",
        "        f.write(uploaded_file.read())\n",
        "\n",
        "    st.info(\"Processing document...\")\n",
        "\n",
        "    # 1. Load PDF\n",
        "    loader = PyPDFLoader(\"uploaded.pdf\")\n",
        "    documents = loader.load()\n",
        "\n",
        "    # 2. Split text into chunks\n",
        "    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)\n",
        "    docs = text_splitter.split_documents(documents)\n",
        "\n",
        "    # 3. Create embeddings\n",
        "    embeddings = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")\n",
        "\n",
        "    # 4. Store in FAISS\n",
        "    db = FAISS.from_documents(docs, embeddings)\n",
        "    retriever = db.as_retriever(search_kwargs={\"k\": 2})\n",
        "\n",
        "    # 5. Connect to HuggingFace LLM (requires API key)\n",
        "    llm = HuggingFaceHub(\n",
        "        repo_id=\"google/flan-t5-small\",  # small model for demo\n",
        "        huggingfacehub_api_token=st.secrets[\"HUGGINGFACEHUB_API_TOKEN\"],\n",
        "        model_kwargs={\"temperature\": 0.1, \"max_new_tokens\": 256}\n",
        "    )\n",
        "\n",
        "    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)\n",
        "\n",
        "    # 6. Ask questions\n",
        "    query = st.text_input(\"Ask a question about the document:\")\n",
        "    if query:\n",
        "        with st.spinner(\"Thinking...\"):\n",
        "            answer = qa_chain.run(query)\n",
        "        st.success(answer)\n"
      ]
    }
  ]
}