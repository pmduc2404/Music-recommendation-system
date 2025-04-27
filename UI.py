import streamlit as st
from PIL import Image
import torch
import os
from transformers import BitsAndBytesConfig, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from sentence_transformers import SentenceTransformer
import chromadb

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set device based on availability of GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CHROMA_DB_PATH = "./chroma_db"
SONG_COLLECTION_NAME = "song_collection_2019"

# Load models with caching
@st.cache_resource
def load_embedding_model():
    # Create or load the chromadb collection
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.create_collection(name=SONG_COLLECTION_NAME)
    except:
        collection = client.get_collection(name=SONG_COLLECTION_NAME)
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence embedding model
    return embed_model, collection

@st.cache_resource
def load_LLM_model():
    # Load the LLM model based on device
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            use_cache=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, use_cache=True)
    
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    model.load_adapter("./checkpoint_qwen")  # Load adapter if available
    return model, processor

# Initialize models
embed_model, collection = load_embedding_model()
model, processor = load_LLM_model()

############################################################

st.title("Image Captioning with Emotion Analysis")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process image for caption generation
    image_inputs = image.resize((224, 224))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image with emotion."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with st.spinner("Generating caption..."):
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    st.subheader("Generated Caption:")
    caption = output_text[0]
    st.write(caption)

    # Use the generated caption as the query for music recommendation
    st.subheader("Music Recommendations Based on Image Caption:")

    query_embedding = embed_model.encode([caption])[0].tolist()

    # Query the collection for similar items
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    for i, (id, metadata) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
        with st.expander(f"Match {i+1}"):
            st.markdown(f"**Track Name:** {metadata['track_name']}")
            st.markdown(f"**Artist Name:** {metadata['artist_name']}")
            st.markdown(f"**Genre:** {metadata['genre']}")
            st.markdown(f"**Topic:** {metadata['topic']}")


