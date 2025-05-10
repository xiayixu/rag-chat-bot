import requests
import pandas as pd
import json
import time
from tqdm import tqdm

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

def save_to_csv(questions, answers, filename='qa_test.csv'):
    data = {'Question': questions, 'Answer': answers}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"answer saved to {filename}")

qa_list=['Who is Kyle',
         'Who are our customers?',
         'How to Fine-Tune a Multiline Model and what is the link?',
         'What is the Ip address of Alpha12?',
         'How to Fine-Tune an OCR Model?',
         'what are latest image for opp?',
         'who are the contact persons for all our customers?',
         'what were the steps to do tangoe upgrades?',
         'How to Fine-Tune an OCR Model?',
         'what is the disaster recovery policy?',
         'please list all our customer contract dates',
         'how many languages our OCR support?',
         'tell me about AI Pathfinder\n is AI Pathfinder a multimodal model?  if yes, what are the different modalities?'
         ]

answers = []
for qa in tqdm(qa_list):
    print(f'qa:{qa}')
    res = ask_response(qa)
    qa_answer = res.json()["official_response"]
    answers.append(qa_answer)
    ask_response('clear')
    print('question is answered')
    time.sleep(3)

filename = 'qa_with_reranker_v2.csv'
save_to_csv(qa_list, answers,filename)