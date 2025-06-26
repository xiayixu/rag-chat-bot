import streamlit as st
import requests
import json
import time
import base64
from pathlib import Path
import fitz
import os
import shutil
from PyPDF2 import PdfMerger,PdfReader,PdfWriter 
from tqdm import tqdm


#get respones through Flask
def ask_response(prompt):
    url = "http://192.168.1.168:2024/chained_rag"

    payload = json.dumps({
    "prompt": prompt
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response

from atlassian import Confluence
from io import BytesIO
#download pdf page
def download_pdf(dic):
    confluence = Confluence(
    url='<url for confluence>',
    username='<username for confluence>',
    password='<password for confluence>',
    api_version='cloud',
    cloud=True)

    Unq_dic = list({frozenset(item.items()): item for item in dic}.values())

    output_path = '/data2/yixu/semantic_search/port/temporary'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    pdf_writer = PdfWriter()
    for page in tqdm(Unq_dic):
        page_id = page["page_id"]
        pdf_byte = confluence.export_page(page_id=page_id)
        pdf_reader = PdfReader(BytesIO(pdf_byte))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)


    output_file = os.path.join(output_path, 'related data.pdf')
    with open(output_file, 'wb') as output_file_pdf:
        pdf_writer.write(output_file_pdf)
    return output_file


    # writer = PdfWriter()
    # output_path = 'temporary'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    
    # for doc in tqdm(Unq_dic):
    #     doc_reader = PdfReader(doc['source_data'])
    #     doc_page = doc['page_num'] - 1
    #     page = doc_reader.pages[doc_page]
    #     writer.add_page(page)
    # output_file = os.path.join(output_path, 'related data.pdf')
    # with open(output_file, 'wb') as output_file_pdf:
    #     writer.write(output_file_pdf)
    # return output_file


#header
st.markdown("""
        # RAG Chained Chat Box
        """)


#message session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "download_links" not in st.session_state:
    st.session_state.download_links = []

if "result_json" not in st.session_state:
    st.session_state.result_json = []

with st.chat_message("Assistant"):
    st.write("Hello 👋")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#Constructing chat Box
if prompt := st.chat_input("i.e tell me about our customer. Enter clear to restart the chat"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


#get question
if prompt:
    st.session_state.download_links = []
    with st.spinner('Waiting for response...'):
        response = ask_response(prompt)

    if response.status_code == 200:
        #clean the chat history
        if response.json()['system'] == 'chat history cleared':
            st.chat_message("Assistant").markdown('restart the chat')
            st.session_state.messages.append({"role": "Assistant", "content": 'restart the chat'})
            shutil.rmtree('/data2/yixu/semantic_search/port/temporary', ignore_errors=True)

        #reload conference dataset
        elif response.json()['system'] == 'All pages are reloaded':
            st.chat_message("Assistant").markdown('All confluence pages are reloading.')
            st.session_state.messages.append({"role": "Assistant", "content": 'restart the chat'})

        #get answers
        else:
            if response.json()["official_response"]:
                off_res = response.json()["official_response"].replace(':',' : ').replace('$','USD ').replace('\n','  \n\n')

                st.chat_message("Assistant").markdown(off_res)
                st.session_state.messages.append({"role": "Assistant", "content": off_res})

            #source ducuments
            if response.json()["source_docs"]:
                st.chat_message("Assistant").markdown("here are the related resource documents")
                download_file = response.json()["result_json"]
                # docs = response.json()["source_docs"]
                # st.write( list({frozenset(item.items()): item for item in docs}.values()))
                # download_doc_path = download_pdf(docs)
                st.session_state.download_links.append(download_file)

        st.session_state.response_data = response.json()
    else:
        st.write("Error: ", response.status_code)


#show the return source json at side bar
if 'response_data' in st.session_state:
    st.sidebar.markdown("#### RESULT JSON")
    st.sidebar.json(st.session_state.response_data)


#provide source code download button
if st.session_state.download_links:
    path = st.session_state.download_links[0]
    if os.path.exists(path):
        with open(path, "rb") as pdf_file:
            # PDFbyte = pdf_file.read()
            st.download_button(label='Download related source data',
                                data=pdf_file,
                                file_name='Download related source data.json',
                                mime='application/octet-stream',
                            #    key=f'download_button_{i}'
                                )
    else:
        st.warning("The file does not exist.")




rebuild_button = st.sidebar.button("reload the confluence data")
if rebuild_button:
    prompt = 'reload confluence'
    user_response = 'All confluence pages are reloading.'
    st.chat_message("Assistant").markdown(user_response)
    st.session_state.messages.append({"role": "Assistant", "content": user_response})

    with st.spinner('Waiting for reloading...'):
        response = ask_response(prompt)
    
        if response.status_code == 200:
            user_response = "All confluence pages are reloaded."
            st.chat_message("Assistant").markdown(user_response)
        else:
            st.write("Error: ", response.status_code)





clear_button = st.button("Start a new question", key="clear_button")
if clear_button:
    prompt = "clear"
    with st.spinner('Clearing chat history...'):
        response = ask_response(prompt)

    if response.status_code == 200 and response.json()['system'] == 'chat history cleared':
        st.chat_message("Assistant").markdown('Chat history cleared.')
        if os.path.exists('/data2/yixu/semantic_search/port/result.json'):
            os.remove('/data2/yixu/semantic_search/port/result.json')

    else:
        st.write("Error: ", response.status_code)
