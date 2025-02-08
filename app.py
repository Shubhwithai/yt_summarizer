from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import streamlit as st
import os

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable

# To run locally:
# 1. Install the required packages:
#     pip install -r requirements.txt
# 2. Run the Streamlit app:
#     streamlit run app.py

# Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "YTUrlLoaded" not in st.session_state:
    st.session_state.YTUrlLoaded = None
if "video_qa_chain" not in st.session_state:
    st.session_state.video_qa_chain = None

# Initialize models
llm = ChatOpenAI(temperature=0.7, model_name='gpt-4')

llama_model = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    openai_api_key=st.secrets["TOGETHER_API_KEY"],
    openai_api_base="https://api.together.xyz/v1"
)

mixtral_model = ChatOpenAI(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    openai_api_key=st.secrets["TOGETHER_API_KEY"],
    openai_api_base="https://api.together.xyz/v1"
)

# Utility Functions
def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    try:
        YouTube(url)
        return True
    except Exception:
        return False

def check_transcript_availability(video_url):
    """Check if a transcript is available for the video."""
    try:
        video_id = YouTube(video_url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return True
    except (TranscriptsDisabled, NoTranscriptAvailable):
        return False
    except Exception as e:
        st.error(f"Error checking transcript availability: {str(e)}")
        return False

def load_video_transcript(video_url):
    """Load the transcript for a YouTube video."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=True,  # Include video metadata
            language="en",  # Default language
            translation="en"  # Optional translation
        )
        data = loader.load()
        
        if not data:
            raise ValueError("No transcript available for this video")
            
        return data
    except Exception as e:
        raise Exception(f"Failed to load transcript: {str(e)}")

def create_qa_chain(vectorstore, model):
    """Create a conversational QA chain."""
    template = """You are a helpful AI assistant that answers questions about passed context only.
    
    context: {context}

    chat history: {chat_history}
    
    User Query: {question} 
    
    Answer above question in {lang}

    {human_input}"""
    
    prompt = PromptTemplate(
        input_variables=['context', 'chat_history', 'human_input', 'question', 'lang'],
        template=template
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        input_key="human_input",
    )

    video_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    return video_qa_chain

def display_chat_history():
    """Display the chat history."""
    for role, avatar, message, lang in st.session_state.chat_history:
        with st.chat_message(role, avatar=avatar):
            st.write(message)

def process_url(video_url, model):
    """Process the YouTube URL and create a QA chain."""
    try:
        # Validate the YouTube URL
        if not is_valid_youtube_url(video_url):
            st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video link.")
            return False
        
        # Check if a transcript is available
        if not check_transcript_availability(video_url):
            st.error("❌ This video does not have an available transcript. Please try a different video.")
            return False
        
        # Load and process the transcript
        with st.spinner("Processing YouTube URL..."):
            website_data = load_video_transcript(video_url)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            video_splits = text_splitter.split_documents(website_data)

            video_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = FAISS.from_documents(video_splits, video_embeddings)

            if model == "Llama":
                model = llama_model
            elif model == "Mistral":
                model = mixtral_model

            st.session_state.video_qa_chain = create_qa_chain(vectorstore, model)
            st.session_state.YTUrlLoaded = True

        st.success("✅ Processing complete! You can now ask questions about the video.")
        return True
        
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        return False

def initiate_processing():
    """Reset the chat history and process the URL."""
    st.session_state.YTUrlLoaded = False
    st.session_state.chat_history = []

    if st.session_state.video_url:
        process_url(st.session_state.video_url, st.session_state.model)

# Frontend Code
st.set_page_config(page_title="Video Summarizer", page_icon="🎥")

# Sidebar for URL input
with st.sidebar:
    st.subheader("Your YouTube URL")
    
    st.text_input(
        label="Youtube Url",
        type="default",
        placeholder="Enter any YouTube video url",
        disabled=False,
        key="video_url",  # Store key in session state
        on_change=initiate_processing
    )

    st.selectbox(
        "Select AI Model",
        ("Mistral", "Llama"),
        key="model",
        on_change=initiate_processing
    )

# Main chat interface
st.header("Video Summary 📽")
st.subheader("Ask questions about the video or Summarize in any Language")

if st.session_state.YTUrlLoaded:
    user_question = st.chat_input("Ask a Question or Summarize")
    
    lang = st.text_input(
        label="Language",
        value="English",
        max_chars=15,
        type="default",
        disabled=False
    )

    if user_question:
        user_role = "User"
        user_avatar = "👩‍🦰"

        # Add question to chat history
        st.session_state.chat_history.append((user_role, user_avatar, user_question, lang))

        # Display chat history
        display_chat_history()

        try:
            with st.spinner("Thinking..."):
                response = st.session_state.video_qa_chain({
                    "human_input": '',
                    "question": user_question,
                    "lang": lang
                })
                assistant_role = "Teacher"
                assistant_avatar = "👩‍🏫"
                st.session_state.chat_history.append((assistant_role, assistant_avatar, response["answer"], lang))
                
                # Display the assistant's response
                with st.chat_message(assistant_role, avatar=assistant_avatar):
                    st.write(response["answer"])
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

# Display initial instructions
else:
    st.write("👈 Enter any YouTube url in the sidebar to get started!")
