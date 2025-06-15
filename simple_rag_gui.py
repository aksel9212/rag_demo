import os, re, time
import numpy as np
import asyncio

from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
import google.generativeai as genai

import streamlit as st
from pypdf import PdfReader, PdfWriter
#from PyPDF2 import PdfReader, PdfWriter
import fitz 
import base64
import io 
import threading
from typing import Union
import av
import PIL 
import cv2
from collections import defaultdict
#from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

WORKING_DIR = "rag-data"
DOCS_DIR = os.path.join(WORKING_DIR, "docs")
EMBS_DIR = os.path.join(WORKING_DIR, "embeddings")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(DOCS_DIR):
    os.mkdir(DOCS_DIR)
if not os.path.exists(EMBS_DIR):
    os.mkdir(EMBS_DIR)
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
openai_api_key = st.secrets.openai_api_key #os.getenv("GROQ_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

ai_provider = st.secrets.ai_provider

if ai_provider == 'openai':
    emb_model = None
else:
    if "emb_model" not in st.session_state or not st.session_state.emb_model:
        emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512)

#additional_info = None
last_paragraph = None

describe_image_prompt = """
Anbei ein Foto vom Typenschild eines Geräts. Können Sie daraus bitte den genauen Gerätetyp entnehmen?
Hier ist das Bild:
"""
def embedding_func(texts: list[str]) -> np.ndarray:
    print("text to embeddings:",texts)
    if emb_model:
        embeddings = emb_model.encode(texts, convert_to_numpy=True,show_progress_bar=True)
        return embeddings
    return None

def openai_embedding(texts: list[str]) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [dat.embedding for dat in response.data]

def llm_model_func_google(prompt, history_messages=[]) -> str:
    print(prompt)
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    
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

def llm_model_image_func_google(prompt=describe_image_prompt,images=[],binaries=False) -> str:
    
    combined_prompt = [prompt]
    if binaries :
        combined_prompt += images
    else:
        combined_prompt += [PIL.Image.open(image) for image in images] 
    # Call the Gemini model
    response = model.generate_content(combined_prompt)
    print(response.text)
    time.sleep(10)
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
    gen_llm = None#llm_model_func_openai
    embedding_model = openai_embedding
elif ai_provider == 'google':
    gen_llm = llm_model_func_google
elif ai_provider == 'deepseek':
    gen_llm = llm_model_func_deepseek

def load_embs(embs):
    embeddings_path = os.path.join(EMBS_DIR,embs)
    if os.path.exists(embeddings_path):
        e = np.load(embeddings_path)
        return e
    return []

def save_embs(embs,embeddings_file):
    embeddings_path = os.path.join(EMBS_DIR,embeddings_file)
    np.save(embeddings_path,embs)

def load_file(filename):
    index_path = os.path.join(WORKING_DIR,filename)
    if os.path.exists(index_path):
        with open(index_path,"r") as f:
            index = [doc.strip() for doc in f.readlines() if doc.strip()]
        return index
    return []

def get_pdf_text(pdf_doc):

    pages_text = []
    
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text = page.extract_text()
        pages_text.append(text)
    return pages_text

def run_new_indexing():
    
    def process_pdf_pages(page):
        
        #global additional_info
        global last_paragraph

        prompt = """
        You are a highly capable and accurate language model skilled in pdf text recognition, analysis, and description generation. 
        You are given an image of a pdf file page. 
        
        ### Instructions:

        Step 1:
           - Analyze the provided pdf page, extract all visible text except the header and the footer of the page.
           - The header and the footer of the page must be skipped.
           - Make a comprehensive description for figures (figure title: description). 
           - For tables, create an independent paragraph that provides a short overview of the table's content. This overview should be concise, informative and prefixed by the title of the table.    
        
        Step 2:
           - Organize the content into semantically meaningful paragraphs based on themes. If there are two or more distinct themes in the content, create separate paragraphs for each theme.
           - Use the "Last paragraph from the previous page" as additional context to better undedrstand the current page content.
           - Do not add  the "Last paragraph from the previous page" section to the output.
           - If the table's content spans multiple paragraphs, include the mention [from table table_title] in each paragraph to indicate its source. Replace "table_title" with the actual title of the table.
        
        Step 3:
            - Enclose each paragraph within <PARAGRAPH> and </PARAGRAPH> tags for clear separation.
            - Each paragraph should be formulated to ensure maximum compatibility with retrieval-augmented generation (RAG) systems. 
            - Use concise, clear language and focus on delivering context-rich, standalone paragraphs.
            - All output must be in German language. 

        ### Last paragraph from the previous page: {last_p}

        ### Here is the image: 
        """
        response = llm_model_image_func_google(prompt.format(last_p=last_paragraph),[page])
        pattern = r"<PARAGRAPH>(.*?)</PARAGRAPH>"
        response = re.findall(pattern, response, re.DOTALL)
        if not response:
            return response

        #if additional_info:
        #    response = response[:1] + response

        #if '[INCOMPLETE]' in response[-1]:
        #    additional_info = response[-1].replace('[INCOMPLETE]','')
        #    response = response[:-1]
        #else:
        #    additional_info = None
        last_paragraph = response[-1]
        return [p.strip() for p in response]
    
    def split_text_doc():
        pass
    
    documents = os.listdir(DOCS_DIR)
    new_documents = {}
    for doc in documents:

        # skip already indexed docs
        if doc in st.session_state.indexed_docs:
            continue    
        # index a new doc
        print(f"indexing {doc}")
        doc_path = os.path.join(WORKING_DIR,doc)
        # create a new dir for the new doc
        if not os.path.exists(doc_path):
            os.mkdir(doc_path)

        # if the doc is a pdf
        if os.path.splitext(doc)[1].lower() == '.pdf':
            # extract pages content
            if len(os.listdir(doc_path)) == 0: 
                pdf_pages = fitz.open(os.path.join(DOCS_DIR,doc))
                pdf_pages = [io.BytesIO(page.get_pixmap().tobytes("png")) for page in pdf_pages]
                pdf_pages = [process_pdf_pages(page) for page in pdf_pages]
                curr_docs_data = [part for page in pdf_pages for part in page]
                docs_names = [doc + f" (page {i+1} part {j+1})" for i in range(len(pdf_pages)) for j in range(len(pdf_pages[i]))]
                for doc_idx in range(len(curr_docs_data)):
                    with open(os.path.join(doc_path,f"{doc_idx}#{docs_names[doc_idx]}"),'w', encoding="utf-8") as fp:
                        fp.write(curr_docs_data[doc_idx]) 

            else:
                curr_docs_data = []
                docs_names = sorted(os.listdir(doc_path), key=lambda x: int(x.split('#')[0]))
                
                for file in docs_names:
                    with open(os.path.join(doc_path,file),'r',encoding="utf-8") as fp:
                        curr_docs_data.append(fp.read())


            # get all document pages and all chunks for every page, save the chunks in separate files inside the 
            # document folder
             
            
            # create doc abstract
            prompt = """
            ### Instructions:

            1- Analyze the following texts, which represents the first 10 pages of a PDF document.
            2- Based on this initial section, please provide a detailed description of the document's content.

            ### Output format:

            - Please enclose your response between <ABSTRACT> and </ABSTRACT> and structure it with the following sections:
            
            <ABSTRACT>
            [what this document is about?]

            [Who do you think this document is written for? (e.g., academics, students, general public, professionals in a specific field).]
            </ABSTRACT>
            
            - The output must not exxeed 512 tokens 
            - The output must be in German language. 

            ### Input 
            Here is the text from the first 10 pages:
            {pages}

            """
            first_pages = curr_docs_data[:30] 
            response = llm_model_func_google(prompt.format(pages="\n\n".join(first_pages)))
            pattern = r"<ABSTRACT>(.*?)</ABSTRACT>"
            response = re.findall(pattern, response, re.DOTALL)
            new_documents[doc] = response[0]
            
            
        # process text documents
        else:
            with open(doc_path,'r') as f:
                curr_docs_data = split_text_doc(f.read())
                docs_names = [doc + f"part {i+i}" for i in range(len(doc_parts))]
                for doc_idx in range(len(curr_docs_data)):
                    with open(os.path.join(doc_path,f"{doc_idx}#{docs_names[doc_idx]}"),'w', encoding="utf-8") as fp:
                        fp.write(curr_docs_data[doc_idx]) 
        
        # update the indexed documents
        st.session_state.indexed_docs.append(doc)
        # create document's embeddings
        if len(curr_docs_data):
            embeddings = openai_embedding(curr_docs_data)
            save_embs(embeddings,f"{doc}_embeddings")
    

    # embed the descriptions of the new documents
    print("generate embeddings....")
    if len(new_documents):
        embeddings = openai_embedding(list(new_documents.values()))
        if len(st.session_state.indexed_docs_embs) > 0:
            st.session_state.indexed_docs_embs = np.vstack([st.session_state.indexed_docs_embs,embeddings])         
        else:   
            st.session_state.indexed_docs_embs = embeddings
        save_embs(st.session_state.indexed_docs_embs,"indexed_docs_embeddings")
    
    print("write index.....")
    with open(os.path.join(WORKING_DIR,'indexed_docs.txt'),"w") as f:
        f.write("\n".join(st.session_state.indexed_docs))
    print("Documents indexing DONE!!")


def save_docs(docs):
    for doc in docs:
        file_path = os.path.join(DOCS_DIR, doc.name)
        with open(file_path ,'wb') as fp:
            fp.write(doc.getbuffer())



def remove_file_data(file_to_remove):
    try:
        # Read the filenames from index.txt
        index_file = os.path.join(WORKING_DIR,'indexed_docs.txt')
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
        embeddings_file = os.path.join(EMBS_DIR,'indexed_docs_embeddings.npy')
        embeddings = np.load(embeddings_file)
        if remove_index >= len(embeddings):
            print("Error: Index out of range for embeddings.")
            return

        # Remove the embedding at the specified index
        embeddings = np.delete(embeddings, remove_index, axis=0)
        print(f"Removed embedding at index {remove_index} from {embeddings_file}.")
        
        # Save the updated embeddings back to embeddings.npy
        np.save(embeddings_file, embeddings)
        
        
        # remove doc folder
        doc_dir = os.path.join(WORKING_DIR,file_to_remove)
        if os.path.exists(doc_dir):
            for entry in os.scandir(doc_dir):
                os.remove(entry.path)
            os.rmdir(doc_dir)
        
        # remove doc embeddings 
        embs_path = os.path.join(EMBS_DIR,f"{file_to_remove}_embeddings.npy")
        if os.path.exists(embs_path):
            os.remove(embs_path)

        # remove the doc
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



def rag(question):

    def cosine_similarity(vecs_db,query,thredshold=0.40):
        vec1 = np.array(vecs_db)
        vec2 = np.array(query).T
        print(vec1.shape)
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Compute cosine similarities
        similarities = np.dot(vec1_norm, vec2_norm.flatten())
        try:
            print(similarities[74])
            print(similarities[82])
        except:
            pass
        scores =  sorted([score for score in similarities if score > thredshold],reverse=True)
        if len(scores) == 0:
            scores = [max(similarities)]
        idxs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:len(scores)]
        return idxs
    
    
    # embed query
    query = openai_embedding(question)
    # get relevent documents
    docs_idxs = cosine_similarity(st.session_state.indexed_docs_embs,query,0.5)
    
    relevent_docs = [st.session_state.indexed_docs[doc] for doc in docs_idxs]
    docs_chunks_number = []
    embeddings = []
    for doc in relevent_docs:
        doc_embeddings = load_embs(f"{doc}_embeddings.npy")
        docs_chunks_number.append(len(doc_embeddings))
        embeddings.append(doc_embeddings)
    embeddings = np.concatenate(embeddings)

    # get relevent chunks from relevent docs
    chunks_idxs = cosine_similarity(embeddings,query)

    # reverse docs search
    doc_chunks = {doc: [] for doc in relevent_docs}

    ranges = []
    start = 0
    for num_chunks in docs_chunks_number:
        end = start + num_chunks
        ranges.append(range(start, end))
        start = end

    for idx in chunks_idxs[:50]:
        for doc, doc_range in zip(relevent_docs, ranges):
            if idx in doc_range:
                relative_index = idx - doc_range.start
                doc_chunks[doc].append(relative_index)
                break

    return doc_chunks

def load_index():
    st.session_state.indexed_docs_embs = load_embs("indexed_docs_embeddings.npy")
    st.session_state.indexed_docs = load_file('indexed_docs.txt')

def main():
    
    st.title("RAG AI Agent (demo)")
    if "indexed_docs" not in st.session_state:
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
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄{filename}")
                with col2:
                    if st.button("✖", type='tertiary',key=f'btn-{filename}'):
                        remove_file_data(filename)
        
        st.button("",type='tertiary',key='s1')
        if st.button("Update Index"):
            save_docs(new_docs)
            run_new_indexing()
        st.button("",type='tertiary',key='s2')
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
            
            # get best scored doc's chunks 
            docs = rag(user_input)
            print(docs)
            #get chunks contents 
            chunks_paths = []
            for doc,chunks in docs.items():
                chunks_files = os.listdir(os.path.join(WORKING_DIR,doc))
                for chunk in chunks:
                    filename = next((file for file in chunks_files if file.startswith(f"{chunk}#")), None)
                    chunks_paths.append(f"{WORKING_DIR}/{doc}/{filename}")
            
            docs_content = []
            for doc_path in chunks_paths:
                with open(doc_path,"r",encoding="utf-8") as fp:
                    doc_ref = doc_path.split("#")[-1]
                    docs_content.append(f"<SRC>{doc_ref}</SRC>\n" + fp.read())
            
            prompt = """
                You are an expert at analyzing and filtering information. Your task is to select the most relevant and useful documents from a provided list that directly contribute to answering a given question. 

                ### Instructions:
                1. Carefully review each document.
                2. Identify and select the documents that are highly relevant and provide useful information for answering the question.
                3. Only include documents that contribute directly to answering the question.
                4. Output each selected documents enclosed within <DOC> and </DOC> tags, preserving their content exactly as provided.

                ### Inputs:
                question:{question}


                Candidate documents:
                {docs}

                ### Output format:
                
                <DOC>
                <SRC> document_source </SRC>
                document_content
                </DOC>
                
            """
            response = llm_model_func_google(prompt.format(question=user_input,docs="\n\n".join(docs_content)))
            pattern = r"<DOC>(.*?)</DOC>"
            best_docs = re.findall(pattern, response, re.DOTALL)
            pattern = r"<SRC>(.*?)</SRC>"
            best_docs_srcs = [src.strip() for src in re.findall(pattern, response, re.DOTALL)]
            best_docs = [best_docs[i].replace(f"<SRC>{best_docs_srcs[i]}</SRC>","") for i in range(len(best_docs))]


            ### Prompt Template:
            prompt = """
            
            ### Instructions

            Using the information provided in the documents, generate a detailed, accurate, and well-structured answer to the question above. Ensure that:  
            1. The response directly addresses the question.   
            2. It avoids speculation or information not present in the provided documents.  
            3. If conflicting information exists in the documents, address the conflict clearly.  
            4. The output must include only the answer to the question, Do not add any other tokens.
            
            ### Question:  
            {question}  

            ### Documents:  
            {docs}  


            """  

            # Final response without the cursor
            response = llm_model_func_google(prompt.format(question=user_input,docs="\n\n".join(best_docs))) 
            
            # clean refs
            references = []
            for ref in best_docs_srcs:
                cleaned_ref = ref.split(" part")[0].strip() + ")"
                if cleaned_ref not in references:
                    references.append(cleaned_ref)

            #display response
            st.markdown(response + "\n\n REFERENCES:<br>"+ "<br>".join(references), unsafe_allow_html=True)
            st.session_state.messages.append({'role':'assistant','content':response})

            # display the pdf pages
            pattern = r"([a-zA-Z0-9\s._-]+?)\s*\(page\s*(\d+)\)"

            # Find all occurrences that match the pattern in the input string.
            # re.findall returns a list of tuples, where each tuple contains the captured groups
            # (filename, page_number_string).
            matches = re.findall(pattern, ", ".join(references))

            # Use defaultdict(list) for cleaner grouping of pages by filename.
            # This automatically initializes a new list for a key if it doesn't exist,
            # preventing the need for explicit 'if filename not in grouped_pages' checks.
            grouped_pages = defaultdict(list)

            # Iterate over each found match.
            for filename, page_str in matches:
                # Clean up the filename by removing any leading or trailing whitespace.
                file_name = filename.strip()
                # Convert the captured page number string to an integer.
                page = int(page_str)
                
                # Append the page number to the list associated with the file_name.
                # defaultdict handles the creation of the list if the key is new.
                grouped_pages[file_name].append(page)

            # Print the final dictionary.
            print(grouped_pages)
            
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



if __name__ == "__main__":
    main()