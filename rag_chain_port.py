import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBILE_DEVICES']='0'

import bs4
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFacePipeline
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import HuggingFaceDatasetLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from ESsearch_bm25 import ElasticSearchBM25Retriever

import torch
from transformers import pipeline, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

import fitz
import time
import subprocess
import platform
from tempfile import NamedTemporaryFile
from colorama import Fore, Style
import json

from atlassian import Confluence
from tqdm import tqdm
from PyPDF2 import PdfMerger
from io import BytesIO
import concurrent.futures

from flask import Flask, request, jsonify

app = Flask(__name__)

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError,SerializationError
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
import re
#################################################################
# Setup for elasticsearch
#################################################################

def document_to_dict(doc, doc_id):
    return {
        'page_content': doc.page_content,
        'metadata': doc.metadata
    }

def index_documents_bulk(es_client, index_name, documents):
    def generate_actions(docs):
        for doc in docs:
            doc_id = f"{doc.metadata['source_data']}_{doc.metadata['page_id']}__{hash(doc.page_content)}"
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc_id,
                "_source": document_to_dict(doc, doc_id)
            }

    chunk_size = 500 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(documents), chunk_size):
            chunk = documents[i:i + chunk_size]
            futures.append(executor.submit(bulk, es_client, generate_actions(chunk)))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"got error when upload: {e}")

#################################################################
# clean the data
#################################################################
def clean_text(text):
    text = text.replace('\xa0', ' ')
    # text = ' '.join(text.split())
    # pattern = re.compile(r'[\u4e00-\u9fff]+')
    # cleaned_text = re.sub(pattern, '', text)
    return text
#################################################################
# Build retriever
#################################################################
from HTMLloader import BSHTMLLoader

def build_chunks(folder_path):
    all_split_docs = []
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    
    text_splitter = CharacterTextSplitter(
        separator=".", 
        chunk_size=1600, 
        chunk_overlap=256,
        length_function=len,
        is_separator_regex=False,
    )

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.html'):
            html_path = os.path.join(folder_path, filename)
            loader = BSHTMLLoader(html_path)
            datas = loader.load()
            docs = []
            for data in datas:
                text = clean_text(data.page_content)
                source = data.metadata['source']
                page_id = data.metadata['page_id']
                chinese_count = len(chinese_pattern.findall(text))
                total_count = len(text)
                if total_count > 0:
                    chinese_ratio = chinese_count / total_count
                else:
                    chinese_ratio = 0
                
                if chinese_ratio <= 0.8:
                    docs.append(Document(page_content=text, metadata={'source': source, 'page_id': page_id}))

            
            for page in docs:
                split_page = text_splitter.split_documents([page])
                
                for chunk in split_page:
                    page_id = page.metadata['page_id']
                    source = page.metadata['source']
                    chunk.metadata = {'source_data': source, 'page_id': page_id}
                    all_split_docs.append(chunk)
    
    print(f"Total chunks from all PDFs: {len(all_split_docs)}")
    return all_split_docs


def build_retriever(split_docs_with_pages, sentence_transformer_path,reranker_model_path):
    '''
    Build a retriever using the embeddings extracted from an input PDF.

    Inputs
    pdf_path: file path of input PDF (/data2/yixu/semantic_search/port/Confluence_All_pages.pdf)
    sentence_transformer_path: file path to the embedding model (/data2/lmm_dev/models/all-MiniLM-l6-v2)

    Outputs:
    The generate retriever
    '''

    ##Key for local elastic search dataset
    ELASTIC_PASSWORD = "rBVbhRsWdWJ5iI8+dRIf"
    CERT_FINGERPRINT = "fe2c26245fae3bd828898e7cfe976a64182e448e8feb6b596b1477b78a161dd5"
    es = Elasticsearch(
        "https://localhost:9200",
        ssl_assert_fingerprint=CERT_FINGERPRINT,
        basic_auth=("elastic", ELASTIC_PASSWORD)
    )

    index_name = 'chunks'
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        time.sleep(10)
        print(f"Index '{index_name}' reloaded.")
    else:
        print(f"Index '{index_name}' does not exist. No action taken.")
    
    index_documents_bulk(es_client=es, index_name='chunks', documents=split_docs_with_pages)
    
    model_kwargs = {'device':'cuda:1'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=sentence_transformer_path,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    ESretriever = ElasticSearchBM25Retriever(client=es, index_name="chunks")

    db = FAISS.from_documents(split_docs_with_pages, embeddings)
    retrievers = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    
    ensemble_retriever = EnsembleRetriever(retrievers=[ESretriever, retrievers], weights=[0.5, 0.5])

    re_rank_model_name = reranker_model_path #"/data2/yixu/bge-reranker-v2-m3"
    model_kwargs = {
                    'device': 'cuda:1', 
                    'trust_remote_code':True,
                    }
    model = HuggingFaceCrossEncoder(model_name=re_rank_model_name, 
                                    model_kwargs = model_kwargs,
                                    )
    compressor = CrossEncoderReranker(model=model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    return compression_retriever

#################################################################
# Initiate LLM
#################################################################

def initialize_llm(llm_model_path, max_new_tokens=768, max_length=4096):
    '''
    Inputs


    llm_model_path: /data2/lmm_dev/models/Mistral-7B-Instruct-v0.2
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # quantization_config = BitsAndBytesConfig(
    #     llm_int8_enable_fp32_cpu_offload=True,  
    #     bnb_8bit_compute_dtype=torch.float16, 
    #     load_in_8bit=True 
    # )


    quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_4bit=True
        )
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, 
                                                quantization_config=quantization_config, 
                                                trust_remote_code=True
                                                )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)


    
    text_generation_pipeline = pipeline("text-generation", 
                                        model=model,
                                        tokenizer=tokenizer, 
                                        max_new_tokens=max_new_tokens,
                                        device_map="auto",
                                        )

    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        model_kwargs={"temperature": 0.2, "max_length": max_length,'top_k':50},
    )
    return llm

#################################################################
# Contextualize question 
#################################################################
def contextualize_retriever(llm, retrievers):
    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. \
    Do NOT answer the question. \
    Do not generate irrelevant questions. If the latest question is already standalone, return it as is. \
    Do not add any additional roles, only include Human and Assistant. \
    The return should always be a single standalone question."""
    # contextualize_q_system_prompt = """Given a chat history and the latest user question \
    # which might reference context in the chat history, formulate a standalone question \
    # which can be understood without the chat history. Do NOT answer the questions, \
    # just reformulate it if needed and otherwise return it as is.\
    # Do not add any additional roles, only include Human and Assistant. """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retrievers, contextualize_q_prompt
    )

    return history_aware_retriever

#################################################################
# Answer question 
#################################################################
def create_rag_chain(llm, retrievers):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Answer my question with the provided context below. \
    Ignore any information that is irrelevant to the answer and only answer tha latest question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer brief,detailed and no more than ten sentences. \
    Make sure your answers are easy to read and understand.
    Your response must be within 2000 characters. \
    Do not repeat the question in your answer. \
    Do not generate any other roles besides Human and Assistant. \
    Do not generate new questions. \
    The return should always be a single QA pair in the following format: \

    Human: [Question] \
    Assistant: [your answer] \

    Ensure the output strictly follows this format without any deviations. Do not generate multiple sets of questions and answers. \
        
    {context} \
    You should answer the question with the above information and ignore any irrelevant details. \
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retrievers, question_answer_chain)
    return rag_chain

#################################################################
# Find the correct question in the answer chain
#################################################################

def similarity(str1,str2):
    vectorizer = CountVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)
    simi = cosine_sim[0][1]
    return simi

def find_question(prompt,filtered_matches):
    last_index = -2
    for i in range(len(filtered_matches)):
        if i % 2 == 0:
            simi = similarity(filtered_matches[i][1],prompt)
            if simi > 0.86:
                last_index = i
                break
    return last_index

#################################################################
# Get data through API
#################################################################
from export import export_page_as_html
from bs4 import BeautifulSoup


def download_all_confluence_page(output_folder):
    atlassian_site = 'singularity-systems'
    confluence = Confluence(
        url='https://singularity-systems.atlassian.net',
        username='dantanlianxyx@gmail.com',
        password='ATATT3xFfGF0aVaZRzbU5KjDyEJgpDmCY8_b7CkThNxBWU2xoPeduFwMgLHWlrBSixWpGWwxAGbMsos7Pll-FVr3xcDek09WqmjxYoISXDDYo5ogcdUV2dsBvIKH4SSfAiv32zVyl-RsAQmcVlHw7RfwGjrbCq4TdurGo9QcCSza1l74wLiU5Sc=F2584998',
        api_version='cloud',
        cloud=True)

    spaces = confluence.get_all_spaces(start=0, limit=500, expand=None)['results']
    list_of_spaces = []
    for i in spaces:
        if i['type'] != 'personal' and i['key'] != 'JohnnyDeci':
            list_of_spaces.append(i['key'])

    list_of_pages = []
    for space in tqdm(list_of_spaces):
        try:
            progress_bar = tqdm(ncols=100, dynamic_ncols=True)
            print(space)
            count = 0
            while True:
                pages = confluence.get_all_pages_from_space(space= space, start=count, limit=100)
                list_of_pages += pages
                progress_bar.update(1)
                if len(pages) < 100:
                    break
                count += 100
                progress_bar.close()
        except Exception as Argument:
            print('报错：', Argument)
        
    print(f'Total number of pages is {len(list_of_pages)}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file = os.path.join(output_folder, 'download_log.json')

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            download_log = json.load(f)
    else:
        download_log = {}

    for page in tqdm(list_of_pages):
        page_id = page['id']
        title = page['title'].replace('.','').replace('/','')
        version = confluence.history(page_id)['lastUpdated']['when'].replace('.','_')
        file_name = f'id_{page_id}_title_{title}.html'
        output_file = os.path.join(output_folder, file_name)
        parents = confluence.get_page_ancestors(page_id = page_id)
        parent_titles = [item['title'] for item in parents if 'title' in item]
        if parent_titles:
            if len(parent_titles) == 1:
                sentence = f"The page {title} is about {parent_titles[-1]}.\n"
            else:
                sentence = f"The page {title} is about {', '.join(parent_titles[:-1])}, and {parent_titles[-1]}.\n"
        else:
            sentence = ''

        if page_id not in download_log or download_log[page_id] != version:
            html = export_page_as_html(atlassian_site=atlassian_site,page_id=page_id)

            soup = BeautifulSoup(html, 'html.parser')
            new_paragraph = soup.new_tag('p')
            new_paragraph.string = sentence
            body_tag = soup.body
            body_tag.insert(0, new_paragraph)

            with open(output_file,  'w', encoding='utf-8') as html_file:
                html_file.write(str(soup))
            
            download_log[page_id] = version
        else:
            print(f'file {page_id}_{title}_{version} already downloaded')


        with open(log_file, 'w') as f:
            json.dump(download_log, f)

    return output_folder

#################################################################
# Main
#################################################################

#global
retriever = None
llm = None
history_aware_retriever = None
rag_chain = None
chat_history = None

@app.route('/chained_rag', methods=['post'])
def chained_rag():
    output = {}
    global retriever, llm,history_aware_retriever,rag_chain,chat_history

    if not retriever or not llm or not rag_chain:
        folder_path = download_all_confluence_page(output_folder='/data2/yixu/confluence_html_data')
        sentence_transformer_path = '/data2/lmm_dev/models/all-MiniLM-l6-v2'
        llm_model_path = '/data2/lmm_dev/models/Mistral-7B-Instruct-v0.2' # THIS ONE!!
        reranker_model_path ="/data2/yixu/bge-reranker-v2-m3"

        chunk_list = build_chunks(folder_path)
        retriever = build_retriever(chunk_list, sentence_transformer_path,reranker_model_path)
        llm = initialize_llm(llm_model_path)
        history_aware_retriever = contextualize_retriever(llm, retriever)
        rag_chain = create_rag_chain(llm, history_aware_retriever)

    data = request.json
    print('Get request:')
    print(data)

    if data is None:
        return {'error': 'Missing parameter'}, 400
    
    prompt = data.get('prompt')
    if not prompt:
        return {'error': 'Missing prompt'}, 400
    
    if prompt == "clear":
        rag_chain = create_rag_chain(llm, history_aware_retriever)
        chat_history = []
        print('starting a new chat')
        return {'system': 'chat history cleared'}
    
    elif prompt == "reload confluence":
        output_foleder = '/data2/yixu/confluence_html_data'
        pdf_path = download_all_confluence_page(output_foleder)
        sentence_transformer_path = '/data2/lmm_dev/models/all-MiniLM-l6-v2'
        reranker_model_path ="/data2/yixu/bge-reranker-v2-m3"
        llm_model_path = '/data2/lmm_dev/models/Mistral-7B-Instruct-v0.2' # THIS ONE!!
        chunk_list = build_chunks(folder_path)
        retriever = build_retriever(chunk_list, sentence_transformer_path,reranker_model_path)
        llm = initialize_llm(llm_model_path)
        history_aware_retriever = contextualize_retriever(llm, retriever)
        rag_chain = create_rag_chain(llm, history_aware_retriever)
        print('Already reload the confluence pages data')
        return {'system': 'All pages are reloaded'}
    
    else:
        output['system'] = 'Start generation'
        start_time = time.time()
        result = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        end_time = time.time()
        answer = result['answer']
        context = result['context']

        print('============Source Documents================')
        temp = []
        for doc in context:
            doc_i = {}
            page_source = doc.metadata.get('source_data', 'Unknown')
            page_id = doc.metadata.get('page_id', 'Unknown')
            doc_i['source_data'] = (page_source)
            doc_i['page_id'] = (page_id)
            temp.append(doc_i)
            output["source_docs"] = sorted(temp, key = lambda i: i['page_id'])
            print(f'from source Documents:{page_source}')
            print(f"Page {page_id}")
            documents_data = []
    
        for doc in context:
            document_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            documents_data.append(document_dict)
        
        with open('result.json', 'w', encoding='utf-8') as json_file:
            json.dump(documents_data, json_file, ensure_ascii=False, indent=4)
        output["result_json"] = 'result.json'
        print('============RESULT================')
        output['All related prompt and resources'] = answer
        print(answer)

        regex = re.compile(r"(Human|Assistant|AI):\s*(.*?)\s*(?=\b(?:Human|Assistant|AI):|$)", re.DOTALL)
        matches = regex.findall(answer)
        filtered_matches = [(key, value) for key, value in matches if key.lower() in ['system', 'human', 'assistant']][2:]
        print(filtered_matches)
        question_position = -1
        question_position = find_question(prompt,filtered_matches)
        print(f"question's answer position is {question_position}")

        system_response =  filtered_matches[question_position + 1][1]
        chat_history.extend([HumanMessage(content=prompt),system_response])
        chat_history_dict = [{"type": "human", "content": msg.content} if isinstance(msg, HumanMessage) else {"type": "assistant", "content": msg} for msg in chat_history]
        chat_history_json = json.dumps(chat_history_dict)
        output['chat_history'] = chat_history_json

        print('========OFFICIAL RESPONSE================')
        print(system_response)
        output['official_response'] = system_response
        elapsed_time = end_time - start_time
        print(f"Time taken for retrieve: {elapsed_time} seconds")
        return output

if __name__ == "__main__":

    output_foleder = '/data2/yixu/confluence_html_data'
    folder_path= download_all_confluence_page(output_foleder)

    # sentence_transformer_path ='/data2/yixu/gte-Qwen2-7B-instruct' #large lm, have no enough GPU for it
    # sentence_transformer_path = '/data2/yixu/stella_en_400M_v5' # 
    sentence_transformer_path = '/data2/lmm_dev/models/all-MiniLM-l6-v2' ##best behavior
    # sentence_transformer_path = '/data2/yixu/all-MiniLM-L12-v2'
    # sentence_transformer_path = '/data2/yixu/all-mpnet-base-v2'
    reranker_model_path ="/data2/yixu/bge-reranker-v2-m3"
    llm_model_path = '/data2/lmm_dev/models/Mistral-7B-Instruct-v0.2' # THIS ONE!!

    ###test models
    # llm_model_path = '/data2/yixu/Meta-Llama-3.1-8B' 
    # llm_model_path = '/data2/yixu/Meta-Llama-3.1-8B-Instruct' 
    # llm_model_path = '/data2/yixu/Mistral-7B-Instruct-v0.3' 

    # llm_model_path = '/data2/yixu/Phi-3-mini-128k-instruct' #Phi 3 have no enough GPU for it

    chunk_list = build_chunks(folder_path)
    # chunk_list = build_chunks(output_foleder)
    retriever = build_retriever(chunk_list, sentence_transformer_path,reranker_model_path)
    llm = initialize_llm(llm_model_path)
    history_aware_retriever = contextualize_retriever(llm, retriever)
    rag_chain = create_rag_chain(llm, history_aware_retriever)
    chat_history = []

    app.run(debug=False, port=2024, host='0.0.0.0')
