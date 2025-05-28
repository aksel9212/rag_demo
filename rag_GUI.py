import os, json, re, io
import numpy as np
import asyncio
from collections import defaultdict
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
from pypdf import PdfReader, PdfWriter
import base64
from io import BytesIO
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
    st.session_state['emb_model'] = None
else:
    if "emb_model" not in st.session_state or not st.session_state.emb_model:
        st.session_state['emb_model'] = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512)


describe_image_prompt = """
Anbei ein Foto vom Typenschild eines Geräts. Können Sie daraus bitte den genauen Gerätetyp entnehmen?
Hier ist das Bild:
"""
async def embedding_func(texts: list[str]) -> np.ndarray:
    print("text to embeddings:",texts)
    if st.session_state.emb_model:
        embeddings = st.session_state.emb_model.encode(texts, convert_to_numpy=True)
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

def llm_model_image_func_google(images=[],prompt=describe_image_prompt,binaries=False) -> str:
    
    combined_prompt = [prompt]
    if binaries :
        combined_prompt += images
    else:
        combined_prompt += [PIL.Image.open(image) for image in images] 
    # Call the Gemini model
    response = model.generate_content(combined_prompt)
    
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
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def llm_model_image_func_groq(images=[],prompt=describe_image_prompt) -> str:
    
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # or another Groq-supported model like llama3-70b
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(images[0])}",
                        },
                    },
                ],
            }
        ],
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



#class VideoTransformer(VideoTransformerBase):
#    frame_lock: threading.Lock  
#    out_image: Union[np.ndarray, None]

#    def __init__(self) -> None:
#        self.frame_lock = threading.Lock()
        
#        self.out_image = None

#    def transform(self, frame: av.VideoFrame) -> np.ndarray:
#        out_image = frame.to_ndarray(format="bgr24")

#        with self.frame_lock:
            
#            self.out_image = out_image
#        return out_image

async def remove_file_data(file_to_remove):
    try:
        # Read the filenames from index.txt
        index_file = os.path.join(WORKING_DIR,'kv_store_doc_status.json')
        with open(index_file, 'r', encoding="utf8") as file:
            files_ids = json.load(file)
        ids_to_remove = [i for i,j in files_ids.items() if file_to_remove in j['file_path']]
        print(ids_to_remove)
           
        await asyncio.gather(*(st.session_state.rag.adelete_by_doc_id(doc_id) for doc_id in ids_to_remove))     
        file_path = os.path.join(DOCS_DIR,file_to_remove)
        if os.path.exists(file_path):
            os.remove(file_path)
        print("All associated data has been removed successfully.")
        st.rerun()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    
    st.title("RAG AI Agent (demo)")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
        
    with st.sidebar:
        #ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)
        #st.title("Scan a device")
        #if ctx.video_transformer:

        #    snap = st.button("Capture")
        #    if snap:
        #        with ctx.video_transformer.frame_lock:
        #            out_image = ctx.video_transformer.out_image

        #        if out_image is not None:
                    
        #            st.write("Output image:")
        #            st.image(out_image, channels="BGR")
                    #my_path = os.path.abspath(os.path.dirname(__file__))       
                    #cv2.imwrite(os.path.join(my_path, "../Data/"+"filename.jpg"), out_image)

        #        else:
        #            st.warning("No frames available yet.")



        new_docs = st.file_uploader("Upload PDF files here and update index.", accept_multiple_files=True)
        
        all_documents = os.listdir(DOCS_DIR)
        
        st.title("Index content:")
        st.markdown("""
            <style>
                .stVerticalBlock {
                    gap:0;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.container(height=200):
            for filename in all_documents:
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.write(f"📄{filename}")
                with col2:
                    if st.button("✖", type='tertiary',key=f'btn-{filename}'):
                        asyncio.run(remove_file_data(filename))


        st.button("",type='tertiary',key='s1')
        if st.button("Update Index"):
            save_docs(new_docs)
            run_new_indexing()
        st.button("",type='tertiary',key='s2')
        if st.button("Exit"):
            st.write("Good Bye...!")
            st.stop()
    
    
    
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

    user_input = st.chat_input("What do you want to know?")
    st.divider()
    uploaded_image = st.file_uploader("Search by image", type=["jpg", "jpeg", "png"])

    if user_input:
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, width=300 ,caption="Uploaded Image")
            image_description = llm_model_image_func_google([image],binaries=True)
            print(image_description)
            user_input += f"Hier sind weitere Details: {image_description}"
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
            
            
            pattern = r"\[\w+\]\s+(.*?\.pdf)\s+\(page\s+(\d+)\)"
            matches = re.findall(pattern, response)
            grouped_pages = defaultdict(list)
            for match in matches:
                file_name, page = match[0], int(match[1])
                grouped_pages[file_name].append(page)
            
            grouped_pages = dict(grouped_pages)
            if len(grouped_pages): 
                writer = PdfWriter()
                for file, pages in grouped_pages.items():
                    reader = PdfReader(os.path.join(DOCS_DIR,file))
                    total_pages = len(reader.pages)
                    pages = [p - 1 for p in pages if 1 <= p <= total_pages]
                    for page_index in pages:
                        writer.add_page(reader.pages[page_index])
                
                pdf_buffer = io.BytesIO()
                writer.write(pdf_buffer)
                pdf_buffer.seek(0)
                st.session_state["pdf_content"] = base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")
                if st.session_state["pdf_content"]:
                    pdf_display = f'<iframe src="data:application/pdf;base64,{st.session_state["pdf_content"]}" width="650" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
        
                    if st.button("Close PDF Viewer"):
                        st.session_state["pdf_content"] = None
                        st.rerun()


            st.session_state.messages.append({'role':'assistant','content':response})
            print(response)


if __name__ == "__main__":
    if "rag" not in st.session_state:
        st.session_state.rag = asyncio.run(initialize_rag())
        run_new_indexing()
    main()
