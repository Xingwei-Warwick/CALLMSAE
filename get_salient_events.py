# %%
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import re
from argparse import ArgumentParser


def find_event(text):
    event_list = re.findall(r'\((.*?)\)', text, re.DOTALL)

    return event_list 


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3:70b-instruct-q8_0')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--out_dir', type=str, default='summary/')
    parser.add_argument('--in_dir', type=str, default='raw_data/')
    args = parser.parse_args()


    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature, # default 0.8
        num_ctx=2048*4, # set a window size based on the llm
        system='You are a helpful assistant who is an expert in summarizing text.',
        top_p=args.top_p
    ) # try n=3 to get multiple responses

    prompt = ChatPromptTemplate.from_template('Write a summary of the document below using one paragraph. \n\n Document: """{doc_content}"""\n\n Summary:')

    generator = prompt | llm | StrOutputParser()

    event_llm = ChatOllama(
        model=args.model,
        temperature=0., # default 0.8
        num_ctx=2048*4, # set a window size based on the llm
        system='You are a helpful assistant who is an expert in indentifying events in text.',
        top_p=args.top_p
    ) 

    event_prompt = ChatPromptTemplate.from_template('A structured event is something that happened as described in the text. A structured event is represented as a tuple, which consists of actors, a trigger, and objects. Could you list all the structured events in the following article? Example: 1. (John; married; Alice). 2. (Alice; was hired; by Google). \n\n Article: """{summary}"""')

    event_generator = event_prompt | event_llm | StrOutputParser()

    for file in tqdm(listdir(args.in_dir)):
        f_name = file.split('.')[0]
        if isfile(join(args.out_dir, f"{f_name}.json")):
            continue

        out_dict = {}
        with open(join(args.in_dir, f"{f_name}.json"), 'r') as f:
            origin_doc = json.loads(f.read())

        doc_content = origin_doc['text']

        summary = generator.invoke({'doc_content': doc_content})

        out_dict["summary"] = summary

        events = event_generator.invoke({'summary': summary})

        out_dict["events"] = events

        # event_list = find_event(events)

        with open(join(args.out_dir, f"{f_name}.json"), 'w') as f:
            f.write(json.dumps(out_dict, indent=4))


