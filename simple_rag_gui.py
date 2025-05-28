import os
import numpy as np
import asyncio

from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

import streamlit as st
from pypdf import PdfReader
import base64
from io import BytesIO
import threading
from typing import Union
import av
from PIL import Image
import cv2
#from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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

describe_image_prompt = """
Anbei ein Foto vom Typenschild eines Geräts. Können Sie daraus bitte den genauen Gerätetyp entnehmen?
Hier ist das Bild:
"""
async def embedding_func(texts: list[str]) -> np.ndarray:
    print("text to embeddings:",texts)
    if emb_model:
        embeddings = emb_model.encode(texts, convert_to_numpy=True)
        return embeddings
    return None



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
    gen_llm = llm_model_func_openai
    embedding_model = openai_embed
elif ai_provider == 'google':
    gen_llm = llm_model_func_google
elif ai_provider == 'deepseek':
    gen_llm = llm_model_func_deepseek



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
        if doc in st.session_state.index:
            continue

        doc_path = os.path.join(DOCS_DIR,doc)

        if os.path.splitext(doc)[1].lower() == '.pdf':
            pdf_pages = get_pdf_text(doc_path)
            docs_data += pdf_pages
            #citations += [doc_path] * len(pdf_pages)
            citations += [doc + f" (page {i+1})" for i in range(len(pdf_pages))]
        else:
            with open(doc_path,'r') as f:
                docs_data.append(f.read())
                citations.append(doc_path)
        
        #if len(docs_data) > 10:
        #    st.session_state.rag.insert(docs_data,file_paths=citations)
        #    docs_data = []
        #    citations = []
    
    st.session_state.rag.insert(docs_data,file_paths=citations)
    print("Documents indexing DONE!!")


def save_docs(docs):
    
    for doc in docs:
        file_path = os.path.join(DOCS_DIR, doc.name)
        with open(file_path ,'wb') as fp:
            fp.write(doc.getbuffer())



def remove_file_data(file_to_remove):
    try:
        # Read the filenames from index.txt
        index_file = os.path.join(WORKING_DIR,'index.txt')
        with open(index_file, 'r') as file:
            filenames = file.readlines()
        
        # Strip newline characters and clean the list
        filenames = [filename.strip() for filename in filenames]
        
        if file_to_remove not in filenames:
            print(f"'{file_to_remove}' not found in {index_file}.")
            #return
        
        # Find the index of the file to remove
        remove_index = filenames.index(file_to_remove)
        
        # Remove the filename from the list
        filenames.pop(remove_index)
        print(f"Removed '{file_to_remove}' from {index_file}.")
        
        # Save the updated filenames back to index.txt
        with open(index_file, 'w') as file:
            for filename in filenames:
                file.write(filename + '\n')
        
        # Step 3: Remove the corresponding embedding
        embeddings_file = os.path.join(EMBS_DIR,'embeddings.npy')
        embeddings = np.load(embeddings_file)
        
        if remove_index >= len(embeddings):
            print("Error: Index out of range for embeddings.")
            return
        
        # Remove the embedding at the specified index
        embeddings = np.delete(embeddings, remove_index, axis=0)
        print(f"Removed embedding at index {remove_index} from {embeddings_file}.")
        
        # Save the updated embeddings back to embeddings.npy
        np.save(embeddings_file, embeddings)
        
        # Remove the corresponding description
        desc_file = os.path.join(WORKING_DIR,'images_descriptions.txt')
        with open(desc_file, 'r') as file:
            descriptions = file.readlines()
        
        if remove_index >= len(descriptions):
            print("Error: Index out of range for descriptions.")
            return
        
        # Remove the description at the specified index
        descriptions.pop(remove_index)
        print(f"Removed description at index {remove_index} from {desc_file}.")
        
        # Save the updated descriptions back to files_desc.txt
        with open(desc_file, 'w') as file:
            for description in descriptions:
                file.write(description)

        file_path = os.path.join(DOCS_DIR,file_to_remove)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        print("All associated data has been removed successfully.")
        load_index()
        
        st.rerun()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def rag(question,database,thredshold=0.4):

    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    query = embedding_func(question)
    similarities = [cosine_similarity(query, sentence_embedding) for sentence_embedding in database]
    print(similarities)
    scores =  sorted([score for score in similarities if score > thredshold],reverse=True)
    if len(scores) == 0:
        scores = [max(similarities)]
    idxs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:len(scores)]
    # validate results:
    images = [st.session_state.index[i] for i in idxs][:5]
    
    return images

def load_index():
    st.session_state.index_embs = load_embs()
    st.session_state.index = load_file('index.txt')
    st.session_state.descriptions = [desc.split(':',1)[1].strip() for desc in load_file('images_descriptions.txt')]
    

def main():
    
    st.title("RAG AI Agent (demo)")
    if "index" not in st.session_state:
        load_index()
        
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
            response = rag()
            
            # Final response without the cursor
            response = response.replace("###",'')
            st.markdown(response)
            st.session_state.messages.append({'role':'assistant','content':response})
            print(response)


if __name__ == "__main__":
    main()