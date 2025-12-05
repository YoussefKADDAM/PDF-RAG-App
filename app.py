import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

# Pipeline functions
from rag_pipeline import (
    extract_pdf_text,
    split_into_chunks,
    build_vectorstore,
    load_llm,
    summarize_text,
)

# Streamlit Setup
st.set_page_config(page_title="Ask your PDF (Ollama - Free)")
st.header("Ask your PDF 💬📄")
load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# PDF Upload
pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf:
    # 1. TEXT EXTRACTION
    text = extract_pdf_text(pdf)

    if not text:
        st.error("No text found in PDF. Please upload a readable document.")
        st.stop()

    # 2. TEXT SPLITTING
    chunks = split_into_chunks(text)

    # 3. VECTOR STORE (EMBEDDINGS)
    knowledge_base = build_vectorstore(chunks)

    # 4. LOAD OLLAMA LLM
    llm = load_llm()

    # 5. SUMMARY OPTIONS
    st.subheader("📝 Summary Options")
    col1, col2, col3 = st.columns(3)

    short_summary_btn = col1.button("Short Summary")
    detailed_summary_btn = col2.button("Detailed Summary")
    bullet_summary_btn = col3.button("Bullet Summary")

    if short_summary_btn:
        st.subheader("📌 Short Summary")
        st.write(summarize_text(llm, text, mode="short"))

    if detailed_summary_btn:
        st.subheader("📌 Detailed Summary")
        st.write(summarize_text(llm, text, mode="detailed"))

    if bullet_summary_btn:
        st.subheader("📌 Bullet Point Summary")
        st.write(summarize_text(llm, text, mode="bullet"))

    # 6. CHAT WITH YOUR PDF
    st.subheader("💬 Chat With Your PDF")
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=knowledge_base.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
        )

        result = qa({"query": user_question})
        answer = result["result"]
        sources = result["source_documents"]

        # Save chat history
        st.session_state["messages"].append(("user", user_question))
        st.session_state["messages"].append(("assistant", answer))

        # Display chat history
        st.subheader("🗂️ Chat History")
        for role, msg in reversed(st.session_state["messages"]):
            if role == "user":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 AI:** {msg}")

        # Display sources
        st.subheader("📚 Sources Used")
        for i, doc in enumerate(sources):
            page = doc.metadata.get("page", "Unknown")
            st.markdown(f"**Source {i + 1}: Page {page}**")

        st.markdown("---")
