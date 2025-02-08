from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable

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
        video_id = YouTube(video_url).video_id
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
