import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

HF_KEY = st.secrets["HF_KEY"]

st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("This app uses a Hugging Face model for RAG.", icon="📃")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    
    llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_key=HF_KEY,
        context_window=2048,
        max_new_tokens=256,
        temperature=0.7,
        device_map="auto",
    )
    
    # Set up the embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Configure Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20
    
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
        except Exception as e:
            st.warning(
                "Oops, an error occurred. This might be due to server overload or connection issues. "
                "Please try again in a moment.",
                icon="⚠️"
            )
            st.error(f"Error details: {str(e)}")
