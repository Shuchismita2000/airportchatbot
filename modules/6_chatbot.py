import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


# Streamlit app - Add a sidebar and header
st.set_page_config(
    page_title="Changi & Jewel Chatbot",
    page_icon="✈️",
    layout="centered",
)

# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index into LangChain's vector store
retriever = FAISS.load_local("./data/index_folder", embedding_model, allow_dangerous_deserialization=True)


# Initialize the LLM
def load_llm():
    return Ollama(model="llama3.2", temperature=0)

llm = load_llm()

# Build the RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),
    return_source_documents=True
)



# Sidebar with additional information
st.sidebar.title("About This Chatbot")
st.sidebar.write(
    """
    This chatbot is designed to help you find information about **Changi Airport** 
    and **Jewel Changi Airport**. It uses state-of-the-art AI to search a semantic 
    database and provide relevant answers to your questions.
    """
)
st.sidebar.write("### Features:")
st.sidebar.markdown(
    """
    - Discover attractions, dining, and shopping options.
    - Find information about transportation and facilities.
    - Get personalized recommendations based on your queries.
    """
)
st.sidebar.write("### How It Works:")
st.sidebar.markdown(
    """
    1. Enter your question in the text box below.
    2. Click **Submit** to get top results matching your query.
    3. Explore categories for additional insights.
    """
)

# Main Page Design
st.title("🛫 Changi & Jewel Chatbot 🛬")
st.markdown(
    """
    Welcome to the **Changi and Jewel Chatbot**! Start by selecting a question 
    from the list below or enter your own query in the input box.
    """
)


# Example questions
st.markdown("### 💡 **Need inspiration? Try these questions:**")
example_questions = [
    "What are the top attractions at Jewel Changi Airport?",
    "What are the best dining options at Jewel Changi Airport?",
    "Which luxury brands are available at Changi Airport?",
    "How do I get from Terminal 1 to Jewel?",
    "Are there any lounges available at Changi Airport?",
]

# Radio button for prefilled questions
selected_question = st.radio(
    "Select a question:", 
    options=[""] + example_questions,
    index=0
)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# User Input
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("💬 **Ask me anything about Changi Airport or Jewel Changi Airport!**", 
                               value=selected_question if selected_question else "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Query the RAG system
    result = qa_chain({"query": user_input})

    # Generate the bot response
    bot_response = result['result']
    sources = result.get('source_documents', [])
    source_info = "\n".join([f"- {doc.metadata['category']}: {doc.page_content}" for doc in sources])

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    if sources:
        st.session_state.messages.append({"role": "bot", "content": f"**Sources:**\n{source_info}"})

# Display Chat Messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "bot":
        st.markdown(f"**Bot:** {msg['content']}")


# Footer
st.markdown(
    """
    ---
    **Note:** This chatbot is powered by advanced AI embeddings and FAISS for semantic search. 
    Feel free to explore and discover the wonders of Changi and Jewel Changi Airport!
    """
)
