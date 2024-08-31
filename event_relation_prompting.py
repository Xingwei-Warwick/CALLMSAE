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


REL_MAP = {'hierarchical': 'is_subevent_of', 'temporal': 'happened_before', 'causal': 'caused_by'}


def get_grades(generator, edge_list, document, rel='is_subevent_of'):
    if rel == 'is_subevent_of':
        rel = 'is_a_subevent_of'
    this_relation = rel.replace('_', ' ')
    input_list = []
    for event1, event2 in edge_list:
        generation = f'Event "{event1}" {this_relation} event "{event2}".'
        input_list.append({"documents": document, "generation": generation})

    response_list = generator.batch(input_list)
    return response_list


def find_code(test):
    code_list = re.findall(r"```(.*?)```", test, re.DOTALL)

    if len(code_list) == 0:
        prefix = test.split(')')[:-1]
        test = ')'.join(prefix)
        test += ')\n```'
        code_list = re.findall(r"```(.*?)```", test, re.DOTALL)

    return code_list
        

def find_event_from_code_double(text):
    text = text.replace('\\"', '\'')
    event_list = re.findall(r'\"(.*?)\"', text, re.DOTALL)

    return event_list 


def find_event_from_code_single(text):
    event_list = re.findall(r'\'(.*?)\'', text, re.DOTALL)

    return event_list 


def find_event_from_code(text):
    event_list = find_event_from_code_double(text)
    if len(event_list) == 0:
        event_list = find_event_from_code_single(text)
    return event_list


def find_event_list_id(text):
    event_id_list = re.findall(r'event_list\[(.*?)\]', text, re.DOTALL)

    return event_id_list 


def response_to_edge(response, event_list):
    code = find_code(response)
                
    if len(code) == 0:
        print('No code found')
        return []
    
    code_lines = code[0].split('\n')
    new_code_lines = []
    for line in code_lines:
        if 'event_list' in line and not line.startswith('event_list') and 'for event in event_list' not in line:
            event_id_list = find_event_list_id(line)
            for event_id in event_id_list:
                try: 
                    e_id = int(event_id)
                except:
                    continue
                line = line.replace(f'event_list[{event_id}]', f'"{event_list[e_id]}"')
        new_code_lines.append(line)
    code_lines = new_code_lines

    edges = []
    for i, line in enumerate(code_lines):
        if 'graph.add_edge' in line and line[0]!=' ':
            if line.strip()[-1] == ',':
                event1 = find_event_from_code(line)[0]
                events_in_next_line = find_event_from_code(code_lines[i+1])
                if len(events_in_next_line) > 0:
                    event2 = events_in_next_line[0]
                else:
                    continue
                event2 = find_event_from_code(code_lines[i+1])[0]
            else:
                events = find_event_from_code(line)
                if len(events) >= 2:
                    event1, event2 = events[:2]
                else:
                    # print(events)
                    print(line, len(events))
                    continue

            edges.append((event1, event2))

    return edges


def find_event(text):
    event_list = re.findall(r'\((.*?)\)', text, re.DOTALL)

    return event_list 



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3:70b-instruct-q8_0')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--summary_dir', type=str, default='summary/llama3_train')
    parser.add_argument('--in_dir', type=str, default='temp2')
    parser.add_argument('--out_dir', type=str, default='summary/llama3_train')
    parser.add_argument('--relation', type=str, default='hierarchical')

    parser.add_argument('--use_gold', action='store_true')
    args = parser.parse_args()


    out_dir = f'iterative_outputs/{args.relation}/it'

    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature, # default 0.8
        num_ctx=2048*4,
        system='You are a helpful assistant that can follow instruction to complete Python code',
        top_p=args.top_p
    ) 

    if args.relation == 'temporal':
        prompt = ChatPromptTemplate.from_template('Could you please finish the following code? Do not include the document or the event_list in the final code. Please consider the following hierarchical relations of the events predicted by another agent (it might be wrong) when predicting the temporal relation: {hint_relations} \n\n```python\nimport networkx as nx\n\ndocument = "{doc_content}"\nevent_list = {event_list_str}\n\n# This is a graph representing the temporal relation between the events in the document\n# Each edge in the graph represents a temporal relation between the head and tail nodes which are events\n# An edge means the head event happens before the tail event temporally\n\ntemporal_graph = nx.DiGraph() # This is a directed acyclic graph. There should not be any cycle in the graph. \n\n# Add events as nodes\nfor event in event_list:\n    temporal_graph.add_node(event)\n\n# Add temporal relations as edges to the graph using .add_edge() function. The function takes two strings as inputs. Each string represents an event.\n# There should not be any cycle in the graph. \n# Only add a relation edge if the given document explicitly states the relation or there is strong evidence supporting it. Do not resort to guessing.\n# Explain the reason for each added edge as a comment after each function call\n{existing_edges}```')
    elif args.relation == 'causal':
        prompt = ChatPromptTemplate.from_template('Could you please finish the following code? Do not include the document in the final code. Please consider the following hierarchical and temporal relations of the events predicted by another agent (it might be wrong) when predicting the causal relation: {hint_relations} \n\n\n\n```python\nimport networkx as nx\n\ndocument = "{doc_content}"\nevent_list = {event_list_str}\n\n# This is a graph representing the causal relation between the events in the document\n# Each edge in the graph represents a causal relation between the head and tail nodes which are events\n# An edge means the head event is caused by the tail event. The head event will not happen if the tail event did not happen.\n\ncausal_graph = nx.DiGraph() # This is a directed acyclic graph. There should not be any cycle in the graph. \n\n# Add events as nodes\nfor event in event_list:\n    causal_graph.add_node(event)\n\n# Add causal relations as edges to the graph using .add_edge() function. The function takes two strings as inputs. Each string represents an event.\n# There should not be any cycle in the graph. \n# Only add a relation edge if the given document explicitly states the relation or there is strong evidence supporting it. Do not resort to guessing.\n# Explain the reason for each added edge as a comment after each function call\n{existing_edges}```')
    elif args.relation == 'hierarchical':
        prompt = ChatPromptTemplate.from_template('Could you please finish the following code? Do not include the document or the event_list in the final code. \n\n```python\nimport networkx as nx\n\ndocument = "{doc_content}"\nevent_list = {event_list_str}\n\n# This is a graph representing the hierarchical relation between the events in the document\n# Each edge in the graph represents a subevent relation between the head and tail nodes which are events\n# An edge means the head event is a subevent of the tail event. They are closely related but on different granularity levels.\n\nhierarchical_graph = nx.DiGraph() # This is a directed acyclic graph. There should not be any cycle in the graph. \n\n# Add events as nodes\nfor event in event_list:\n    hierarchical_graph.add_node(event)\n\n# Add hierarchical relations as edges to the graph using .add_edge() function. The function takes two strings as inputs. Each string represents an event.\n# There should not be any cycle in the graph. \n# Only add a relation edge if the given document explicitly states the relation or there is strong evidence supporting it. Do not resort to guessing.\n#Explain the reason for each added edge as a comment after each function call\n{existing_edges}```')

    generator = prompt | llm | StrOutputParser()

    grader_llm = ChatOllama(
            model=args.model,
            temperature=0, # default 0.8
            num_ctx=2048*4, # set a window size based on the llm
            num_predict=64,
        )
    grader_prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Then, provide a short explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )

    hallucination_grader = grader_prompt | grader_llm | StrOutputParser()
    hallucination_grader = hallucination_grader.with_retry()


    if args.relation == 'temporal' or args.relation == 'causal':
        with open('final_iterative_agent_train_hierarchical.json', 'r') as f:
            hier_rel_dict = json.load(f)
        
        hint_str_dict = {}
        for f_id in hier_rel_dict:
            hier_hint_list = []
            for edge in hier_rel_dict[f_id]['relations']:
                hier_hint_list.append(f'Event "{edge[0]}" is a subevent of event "{edge[2]}".')
            hint_str_dict[f_id] = ' '.join(hier_hint_list)
        
        if args.relation == 'causal':
            with open('final_iterative_agent_train_temporal.json', 'r') as f:
                temp_rel_dict = json.load(f)
            
            for f_id in temp_rel_dict:
                temp_hint_list = []
                for edge in temp_rel_dict[f_id]['relations']:
                    temp_hint_list.append(f'Event "{edge[0]}" happened before event "{edge[2]}".')
                hint_str_dict[f_id] += ' '.join(temp_hint_list)
        
        
    # use gold event to generate the graph
    if args.use_gold:
        with open("data/human_annotated_graphs.json", 'r') as f: 
            final_dataset_events = json.loads(f.read())


    not_found = 0
    for file in tqdm(args.in_dir):
        f_id = file.split('.')[0]
        if isfile(f"{out_dir}_0/{f_id}.json"):
            continue

        with open(f"{args.summary_dir}/{f_id}.json", 'r') as f:
            doc_dict = json.loads(f.read())

        if not isfile(f"{args.in_dir}/{f_id}.json"):
            not_found += 1
            print(f_id, 'not found', not_found)
            continue

        with open(f"{args.in_dir}/{f_id}.json", 'r') as f:
            origin_doc = json.loads(f.read())
        doc_content = origin_doc['text'].replace('\n', ' ')

        if args.use_gold:
            event_list = final_dataset_events[f_id]['event_list']
        else:
            events_str = doc_dict['events']
            event_list = find_event(events_str)


        if len(event_list) < 2:
            print(event_list)
            print(f_id, 'too few events')
            continue

        out_dict = {
            'doc_id': f_id
        }

        event_list_str = '["'
        for event in event_list:
            event_list_str += event.replace(';', '').replace('"', '\"') + '", "'
        event_list_str = event_list_str[:-3] + ']'
        
        out_dict['event_list'] = event_list_str

        existing_edges = []

        for k in range(5):
            if args.relation == 'temporal':
                hint_relation_str = hint_str_dict[f_id] if f_id in hint_str_dict else ''
                response = generator.invoke({'hint_relations': hint_relation_str, 'doc_content': doc_content, 'event_list_str': event_list_str, 'existing_edges': "\n".join(existing_edges)})
            elif args.relation == 'causal':
                hint_relation_str = hint_str_dict[f_id] if f_id in hint_str_dict else ''
                response = generator.invoke({'hint_relations': hint_relation_str, 'doc_content': doc_content, 'event_list_str': event_list_str, 'existing_edges': "\n".join(existing_edges)})
            elif args.relation == 'hierarchical':
                response = generator.invoke({'doc_content': doc_content, 'event_list_str': event_list_str, 'existing_edges': "\n".join(existing_edges)})

            out_dict[f"{args.relation}_response"] = response

            edge_list = response_to_edge(response, event_list)

            grades = get_grades(hallucination_grader, edge_list, doc_content, rel=REL_MAP[args.relation])

            out_dict['grades'] = grades

            grade_selection = [True if 'yes' in score.lower() else False for score in grades]

            verified_edges = [edge_list[i] for i in range(len(grade_selection)) if grade_selection[i]]

            num_new_edges = len(verified_edges) - len(existing_edges)

            out_dict['number_of_new_edges'] = num_new_edges
            
            with open(f"{out_dir}_{k}/{f_id}.json", 'w') as f:
                f.write(json.dumps(out_dict, indent=4))

            if k==0 or num_new_edges>0:
                existing_edges = [f'{args.relation}_graph.add_edge("{edge[0]}","{edge[1]}")' for edge in verified_edges]
            elif num_new_edges < 0:
                continue
            else:
                break

            final_edges = [(edge[0], REL_MAP[args.relation], edge[1]) for edge in verified_edges]

            this_save_name = f"final_iterative_agent_{args.relation}.json"
            if isfile(this_save_name):
                with open(this_save_name, 'r') as f:
                    final_dict = json.loads(f.read())
            else:
                final_dict = {}
            
            final_dict[f_id] = {
                'relations': final_edges,
                'iteration': k
            }
            with open(this_save_name, 'w') as f:
                f.write(json.dumps(final_dict, indent=4))



