isimport os
import numpy as np
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed

from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import streamlit as st
from pypdf import PdfReader

import threading
from typing import Union
import av
from PIL import Image
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

WORKING_DIR = "rag-data"
DOCS_DIR = WORKING_DIR + "/docs"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(DOCS_DIR):
    os.mkdir(DOCS_DIR)

# Groq 
groq_api_key = st.secrets.groq_api_key #os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
#deepseek
deepseek_api_key = st.secrets.deepseek_api_key #os.getenv("GROQ_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com")
#gemini
gemini_api_key = st.secrets.google_api_key #os.getenv('GOOGLE_API_KEY') 
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
#openai
openai_api_key =  st.secrets.openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

ai_provider = st.secrets.ai_provider

if ai_provider == 'openai':
    emb_model = None
else:
    emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512)


async def embedding_func(texts: list[str]) -> np.ndarray:
    print("text to embeddings:",texts)
    if emb_model:
        embeddings = emb_model.encode(texts, convert_to_numpy=True)
        return embeddings
    return None


embedding_model = EmbeddingFunc(
            embedding_dim=512,
            max_token_size=8192,
            func=embedding_func,
        )


async def llm_model_func_google(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    
    print("system_prompt:",system_prompt)
    print(prompt)
    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = model.generate_content(combined_prompt)
    print(response.text)
    # 4. Return the response text
    return response.text


async def llm_model_func(prompt,**kwargs) -> str:
    combined_prompt = ""
    
    if 'system_prompt' in kwargs:
        combined_prompt += kwargs['system_prompt'] + f"\n"

        
    if 'history_messages' in kwargs:
        for msg in kwargs['history_messages']:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"


    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or another Groq-supported model like llama3-70b
        messages=[{"role": "user", "content": combined_prompt}],
    )
    return response.choices[0].message.content


async def llm_model_func_deepseek(prompt,**kwargs) -> str:
    print("kwargs:",kwargs)
    combined_prompt = ""
    
    if 'system_prompt' in kwargs:
        combined_prompt += kwargs['system_prompt'] + f"\n"

        
    if 'history_messages' in kwargs:
        for msg in kwargs['history_messages']:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"
    print(combined_prompt)
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": combined_prompt}],
    )

    return response.choices[0].message.content


gen_llm = llm_model_func

if ai_provider == 'openai':
    gen_llm = gpt_4o_mini_complete
    embedding_model = openai_embed
elif ai_provider == 'google':
    gen_llm = llm_model_func_google
elif ai_provider == 'deepseek':
    gen_llm = llm_model_func_deepseek


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gen_llm,
        embedding_func=embedding_model 
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def get_pdf_text(pdf_doc):

    pages_text = []
    
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text = page.extract_text()
        pages_text.append(text)
    return pages_text

def run_new_indexing():
    documents = os.listdir(DOCS_DIR)
    docs_data = []
    citations = []
    for doc in documents:
        doc_path = DOCS_DIR+"/"+doc

        if os.path.splitext(doc)[1].lower() == '.pdf':
            pdf_pages = get_pdf_text(doc_path)
            docs_data += pdf_pages
            #citations += [doc_path] * len(pdf_pages)
            citations += [doc + f" (page {i+1})" for i in range(len(pdf_pages))]
        else:
            with open(doc_path,'r') as f:
                docs_data.append(f.read())
                citations.append(doc_path)
        
        if len(docs_data) > 10:
            st.session_state.rag.insert(docs_data,file_paths=citations)
            docs_data = []
            citations = []
    print(docs_data)
    st.session_state.rag.insert(docs_data,file_paths=citations)
    print("Documents indexing DONE!!")


def save_docs(docs):
    
    for doc in docs:
        file_path = os.path.join(DOCS_DIR, doc.name)
        with open(file_path ,'wb') as fp:
            fp.write(doc.getbuffer())



class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
    
    out_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        
        self.out_image = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        out_image = frame.to_ndarray(format="bgr24")

        with self.frame_lock:
            
            self.out_image = out_image
        return out_image



def main():
    
    st.title("RAG AI Agent (demo)")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"): 
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])    
        
    with st.sidebar:
        ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)
        st.title("Scan a device")
        if ctx.video_transformer:

            snap = st.button("Capture")
            if snap:
                with ctx.video_transformer.frame_lock:
                    out_image = ctx.video_transformer.out_image

                if out_image is not None:
                    
                    st.write("Output image:")
                    st.image(out_image, channels="BGR")
                    #my_path = os.path.abspath(os.path.dirname(__file__))       
                    #cv2.imwrite(os.path.join(my_path, "../Data/"+"filename.jpg"), out_image)

                else:
                    st.warning("No frames available yet.")



        new_docs = st.file_uploader("Upload PDF files here and update index.", accept_multiple_files=True)
        
        all_documents = os.listdir(DOCS_DIR)
        st.markdown(
            """
            <style>
            .custom-scroll-area {
                max-height: 6cm;
                overflow-y: auto;
                background-color: rgba(200, 200, 200, 0.3);  /* light gray with more opacity */
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .custom-scroll-area >p{
                margin-bottom:0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        docs_area = "<div class='custom-scroll-area'>"
        docs_area += f"<h3>Index content:</h3>"
        for doc in all_documents:
            docs_area += f"<p>📄 {doc}</p>"
        docs_area += "</div>"
        st.markdown(docs_area, unsafe_allow_html=True)
    
        
        if st.button("Update Index"):
            save_docs(new_docs)
            run_new_indexing()
        if st.button("Exit"):
            st.write("Good Bye...!")
            st.stop()      
    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({'role':'user','content':user_input})
        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Properly consume the async generator with async for
            response = st.session_state.rag.query(
                        query=user_input,
                        param=QueryParam(mode="local", top_k=5),
            )
            
            # Final response without the cursor
            response = response.replace("###",'')
            st.markdown(response)
            st.session_state.messages.append({'role':'assistant','content':response})
            print(response)


if __name__ == "__main__":
    if "rag" not in st.session_state:
        st.session_state.rag = asyncio.run(initialize_rag())
        run_new_indexing()
    main()
