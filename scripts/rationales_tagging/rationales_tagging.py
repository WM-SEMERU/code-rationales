# %% [markdown]
# # Local Aggregations Module

# %%
def param_default():
    return {
        #'dataset' : 'code_completion_random_cut_5k_30_512_tokens',
        #'dataset' : 'code_completion_docstring_random_cut_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_signature_3.8k_30_150_tokens',
        'dataset' : 'code_completion_docstring_5k_30_150_tokens',
        'rational_results': '/workspaces/code-rationales/data/rationales/gpt',
        'tagged_rationales': '/workspaces/code-rationales/data/tagged_rationales',
        'delimiter_sequence': '', ## VERY IMPOETANT
        'num_samples' : 100, 
        'size_samples' : 157,
        'num_experiments': 30,
        'model_name' : '/workspaces/code-rationales/data/codeparrot-small/checkpoints/checkpoint-29000', 
        'cache_dir': '/workspaces/code-rationales/datax/df_cache_dir',
    }
params = param_default()

# %% [markdown]
# ## CORE

# %% [markdown]
# ### Imports

# %%
from pathlib import Path
import csv
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functools
import json
import nltk
import re

pd.options.display.float_format = '{:.2f}'.format

# %%
from code_rationales.loader import download_grammars
from tree_sitter import Language, Parser
import code_rationales

# %%
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch

# %%
import warnings
import importlib
from matplotlib import colors
import os

# %%
import sys
sys.path.insert(1, '/workspaces/code-rationales/sequential-rationales/huggingface')
from rationalization import rationalize_lm

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ### Setup

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# %% [markdown]
# ### Taxonomies

# %%
#Programming Language Taxonomy
def pl_taxonomy_python() -> dict:
    return {
  "punctuation": ['{', '}', '[', ']', '(', ')','\"', ',', '.', '...', ';', ':'], 
  "exceptions": ['raise_statement','catch', 'try', 'finally', 'throw', 'throws', 'except'],
  "oop": ['def','class','instanceof','interface','private','protected','public','abstract','extends','package','this','implements','import','new','super'],
  "asserts": ['assert'],
  "types": ['tuple','set','list','pair','subscript','type','none','dictionary','integer','native','static','synchronized','transient','volatile','void','final','enum','byte','char','float','boolean','double','int','long','short','strictfp'],
  "conditionals": ['else', 'if', 'switch', 'case', 'default'],
  "loops": ['break', 'do', 'for', 'while', 'continue'],
  "operators": ['as','yield','is','@','in','and','or','not','**','slice','%','+','<','>','=','+','-','*','/','%','++','--','!','==','!=','>=','<=','&&','||','?',':','~','<<','>>','>>>','&','^','|','//'],
  "indentation": ['\n','\t'],
  "bool": ['true', 'false'], 
  "functional":['lambda','lambda_parameters'],
  "with" : ['with','with_item','with_statement','with_clause'], 
  "return" :['return'],
  "structural" : ['attribute', 'argument_list','parenthesized_expression','pattern_list','class_definition','function_definition','block'],
  "statements" : ['return_statement','break_statement','assignment','while_statement','expression_statement','assert_statement'],
  "expression": ['call','exec','async','ellipsis','unary_operator','binary_operator','as_pattern_target','boolean_operator','as_pattern','comparison_operator','conditional_expression','named_expression','not_operator','primary_expression','as_pattern'],
  "errors": ["ERROR"],
  "identifier":["identifier"],  
  "comment":["comment"],
  "string": ['string','interpolation','string_content','string_end','string_start','escape_sequence'], 
  "unknown": []
}

# %%
def nl_pos_taxonomy() -> dict: return {
    "nl_verb" : ['VBN', 'VBG', 'VBZ', 'VBP', 'VBD', 'VB'],
    "nl_noun" : ['NN', 'NNPS', 'NNS', 'NNP'],
    "nl_pronoun" : ['WP', 'PRP', 'PRP$', 'WP','WP$'], 
    "nl_adverb" : ['RBS','RBR', 'RB', 'WRB'], 
    "nl_adjetive" : ['JJR', 'JJS', 'JJ'], 
    "nl_determier" : ['DT','WDT','PDT'], 
    "nl_preposition" : ['IN', 'TO'],
    "nl_particle" : ['RP'],
    "nl_modal" : ['MD'],
    "nl_conjunction" : ['CC'],
    "nl_cardinal" : ['CD'],
    "nl_list": ['LS'],
    "nl_other" : ['FW', 'EX', 'SYM' , 'UH', 'POS', "''", '--',':', '(', ')', '.', ',', '``', '$']
}

# %% [markdown]
# ### AST Mapping

# %%
def unroll_node_types(
    nested_node_types: dict  # node_types from tree-sitter
) -> list: # list of node types
    def iterate_and_unroll_dict(nested_node_types: dict, all_node_types: set):
        for key, value in nested_node_types.items():
            if key == 'type' and type(value) == str:
                all_node_types.add(value)
            if type(value) == dict:
                iterate_and_unroll_dict(value, all_node_types)
            if type(value) == list:
                for element in value:
                    iterate_and_unroll_dict(element, all_node_types) 
    all_node_types = set()
    for dictionary in nested_node_types:
        iterate_and_unroll_dict(dictionary, all_node_types)
    all_node_types.add('ERROR')
    return list(all_node_types)

# %%
def create_parser(lang: str):
    # Grab the node types from the tree-sitter language
    language = Language(f"{code_rationales.__path__[0]}/grammars/tree-sitter-languages.so", lang)
    node_path = f"{code_rationales.__path__[0]}/grammars/tree-sitter-{lang}/src/node-types.json"
    with open(node_path) as f:
            node_types = json.load(f)
    node_types = unroll_node_types(node_types)
    # Create a parser for the language
    parser = Parser()
    parser.set_language(language)
    return parser, node_types

# %%
def traverse(
    node,       # tree-sitter node
) -> None:
    """Traverse in a recursive way, a tree-sitter node and append results to a list."""
    results = []
    def traverse_tree(node, results):
        if node.type == 'string':
            results.append(node)
            return
        for n in node.children:
            traverse_tree(n, results)
        if not node.children:
            results.append(node)
    traverse_tree(node, results)
    return results

# %%
def convert_to_offset(
    point,              #point to convert
    lines: list         #list of lines in the source code
    ):
        """Convert the point to an offset"""
        row, column = point
        chars_in_rows = sum(map(len, lines[:row])) + row
        chars_in_columns = len(lines[row][:column])
        offset = chars_in_rows + chars_in_columns
        return offset

# %%
def get_node_span(node, lines):
    """Get the span position of the node in the code string"""
    start_span = convert_to_offset(node.start_point, lines)
    end_span = convert_to_offset(node.end_point, lines)
    return start_span, end_span
    

# %%
def is_token_span_in_node_span(tok_span, token: str, node_span, node_text: str):
    return (node_span[0] <= tok_span[0] and tok_span[1] <= node_span[1]) or \
            (node_span[0]-1 <= tok_span[0] and tok_span[1] <= node_span[1] and node_text in token)

# %%
def get_token_type(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    token: str,   # token value
    nodes: list,     # list of tree-sitter nodes
    lines: list,     # list of lines in the code
) -> tuple: # (parent_type, token_type) of the token
    """Get the parent AST type and token AST type of a token."""
    for i, node in enumerate(nodes):
        if is_token_span_in_node_span(tok_span, token, get_node_span(node, lines), node.text.decode('utf-8')):
            return nodes[i].parent.type, nodes[i].type

# %%
def get_token_nodes(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    token: str,      #actual token
    node,            # tree-sitter node
    lines: list,     # list of lines in the code
) -> list: 
    """Get all AST types for the given token span"""
    results = []
    def traverse_and_get_types(tok_span, node, lines, results) -> None:
        node_span = get_node_span(node, lines)
        if is_token_span_in_node_span(tok_span, token, node_span, node.text.decode('utf-8')):
            results.append(node)
        for n in node.children:
            traverse_and_get_types(tok_span, n, lines, results)
    traverse_and_get_types(tok_span, node, lines, results)
    return results

# %%
def get_nodes_by_type(
    node, 
    node_types: list
) -> list :
    def traverse_and_search(node, node_types, results):
        if node.type in node_types:
            results.append(node)
        for n in node.children:
            traverse_and_search(n, node_types ,results)
    results = []
    traverse_and_search(node, node_types, results)
    return results

# %% [markdown]
# ### Taxonomy Mapping

# %%
def clean_results(global_results):
    def clean_dictonary(result_dict):
        clean_dict = result_dict.copy()
        for key, value in result_dict.items():
            if not value or not value['values']: 
                clean_dict.pop(key)
        return clean_dict
    for key, value in global_results.items():
        global_results[key] = clean_dictonary(value)
    return global_results

# %%
def search_category_by_token(taxonomy_dict: dict, token_type: str):
    for key, value in taxonomy_dict.items():
        if token_type in value:
            return key
    return 'unknown'

# %%
def map_to_taxonomy(taxonomy_dict: dict, result_dict: dict):
    result_dict = result_dict.copy()
    mappings = {token: {category : {'values': [], 'rationales': []} for category in taxonomy_dict.keys()} for token in result_dict.keys()}
    for target_token, value in result_dict.items():
        for source_token, props in value.items():
            mappings[target_token][search_category_by_token(taxonomy_dict, source_token)]['values'].append(props['values'])
            mappings[target_token][search_category_by_token(taxonomy_dict, source_token)]['rationales'].append(props['rationales'])
    return clean_results(mappings)

# %%
def map_local_results_to_taxonomy(taxonomy_dict:dict, local_results: dict):
    return dict(zip(local_results.keys(), map(lambda aggegrations: map_to_taxonomy(taxonomy_dict, aggegrations), local_results.values())))

# %% [markdown]
# ### Reading Rationales

# %%
get_experiment_path =  lambda samples, size, exp: params['rational_results'] + '/' + params['dataset'] + '/' + '[t_'+str(samples)+']_[max_tgt_'+str(size)+']_[exp:'+str(exp)+'].csv'

# %%
def read_rationales(experiment_paths):
    experiment_results = []
    for exp_idx, experiment_path in enumerate(experiment_paths):
        experiment_result = pd.read_csv(experiment_path, index_col=0)
        experiment_result['exp'] = exp_idx
        experiment_results.append(experiment_result)
    return experiment_results

# %% [markdown]
# ### Rationales Tagging

# %%
calculate_right_span = lambda start_idx, end_idx, df : len(''.join(map(str, df.loc[start_idx:end_idx, 'goal_token'].tolist())))
calculate_span = lambda right_span, token : (right_span-len(str(token)), right_span)
delete_leading_spaces = lambda string: re.sub(r'^\s+', '', string)
delete_leading_breaks = lambda string: re.sub(r'^\n+', '', string)

# %%
def add_first_token_row(df):
    df.loc[-1] = [df['typesets_tgt'][0][0][0], df['from_seq_id'][0], None, None, None, df['exp'][0]]
    df.index = df.index + 1
    df = df.sort_index()
    return df

# %%
def add_auxiliary_columns_to_experiment_result(df, delimiter_sequence: str):
    df.insert(0, 'rational_pos', [i for i in range(len(df))])
    initial_token = df['goal_token'][0]
    ### TOKEN TYPE COLUMN
    token_type_column = ['src'] * len(df)
    sequence = initial_token
    for idx, goal_token in enumerate(df['goal_token']):
        if delimiter_sequence not in sequence:
            token_type_column[idx] = 'nl'
            sequence+=goal_token
    df['token_type'] = token_type_column
    src_initial_token_idx = df[df['token_type'] == 'src'].first_valid_index()
    df['span'] = [None] * len(df[:src_initial_token_idx]) + [calculate_span(calculate_right_span(src_initial_token_idx, index, df), token) for index, token in df[src_initial_token_idx:]['goal_token'].items()]


# %%
def fill_nl_tags_in_experiment_result(df, nl_ast_types, nl_pos_types, parser):
    #initial_token = df['typesets_tgt'][0][0][0] if df[df['token_type'] == 'src'].first_valid_index() == 0 else ''
    ##### POS TAGS FOR NL PART
    target_nl = ''.join(df[df['token_type'] == 'nl']['goal_token'].map(lambda value: str(value)))
    pos_tags = nltk.pos_tag(nltk.word_tokenize(target_nl))
    for idx in range(df[df['token_type']== 'src'].first_valid_index()):
        nl_tags = list(map(lambda tag: tag[1] if tag[1] in nl_pos_types else None, filter(lambda tag: tag[0] in str(df['goal_token'][idx]), pos_tags)))
        if nl_tags: df.at[idx, 'tags'] = df['tags'][idx] + [nl_tags[-1]]
    ##### POS TAGS FOR CODE PART
    target_code = ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    nl_target_nodes = get_nodes_by_type(parser.parse(bytes(target_code, 'utf8')).root_node, nl_ast_types)
    for token_idx in range(df[df['token_type'] == 'src'].first_valid_index(), len(df['span'])):
                for nl_target_node in nl_target_nodes:
                    if is_token_span_in_node_span(df['span'][token_idx], df['goal_token'][token_idx], get_node_span(nl_target_node, target_code.split("\n")), nl_target_node.text.decode('utf-8')) and \
                            (str(df['goal_token'][token_idx]) in nl_target_node.text.decode('utf-8') or nl_target_node.text.decode('utf-8') in str(df['goal_token'][token_idx])):
                            tagged_token_list = list(filter(lambda tagged_token: str(tagged_token[0]).replace(' ','') in str(df['goal_token'][token_idx]).replace(' ','') or str(df['goal_token'][token_idx]).replace(' ','') in str(tagged_token[0]).replace(' ',''), \
                                                        nltk.pos_tag( nltk.word_tokenize(nl_target_node.text.decode('utf-8')))))
                            if len(tagged_token_list)>0 and tagged_token_list[0][1] in nl_pos_types and tagged_token_list[0][1] not in df['tags'][token_idx]: df.at[token_idx, 'tags'] = df['tags'][token_idx] + [tagged_token_list[0][1]]

# %%
def fill_ast_tags_in_experiment_result(df, parser):
    target_code = ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    #target_code = delete_leading_breaks(delete_leading_spaces(target_code))
    src_initial_token_idx = df[df['token_type'] == 'src'].first_valid_index()
    target_ast = parser.parse(bytes(target_code, 'utf8')).root_node
    for token_idx in range(src_initial_token_idx, len(df)):
        df.at[token_idx, 'tags'] = df['tags'][token_idx] + list(map(lambda node: node.type, get_token_nodes(df['span'][token_idx], df['goal_token'][token_idx], target_ast, target_code.split("\n"))))

# %%
def tag_rationals(experiment_results: list, nl_ast_types: list, nl_pos_types: list, delimiter_sequence: str, parser):
    experiments = {}
    for exp_idx, df_experiment in enumerate(experiment_results):
        experiment_results = []
        experiment_rational_results = [df_experiment[(df_experiment['from_seq_id'] == sample_idx) | \
                                                     (df_experiment['from_seq_id'] == str(sample_idx))].reset_index() \
                                                    for sample_idx in range(params['num_samples'])]
        print('*'*10 +'Tagging rationals for exp: ' +str(exp_idx) + '*'*10)
        for experiment_rational_result in experiment_rational_results:
            experiment_rational_result = experiment_rational_result.drop('index', axis=1)
            experiment_rational_result = add_first_token_row(experiment_rational_result)
            add_auxiliary_columns_to_experiment_result(experiment_rational_result, delimiter_sequence)
            experiment_rational_result['tags'] = [[]]*len(experiment_rational_result)
            fill_nl_tags_in_experiment_result(experiment_rational_result, nl_ast_types, nl_pos_types, parser)
            fill_ast_tags_in_experiment_result(experiment_rational_result, parser)
            experiment_results.append(experiment_rational_result)
        experiments[exp_idx] = experiment_results
    return experiments
            

# %% [markdown]
# ### Rationales Aggregation

# %%
def aggregate_rationals(global_tagged_results: dict, ast_node_types: list, nl_pos_types: list, number_samples: int):
    aggregation_results = {sample_id: None  for sample_id in range(number_samples)}
    for exp_idx, experiment_results in global_tagged_results.items():
        print('*'*10 +'Aggregrating rationals for exp: ' +str(exp_idx) + '*'*10)
        for experiment_result in experiment_results:
            ### GET INFORMATION OF FIRST TOKEN
            #sample_results = {str(pos+1)+'['+str(token)+']' : {node_type : {'values': [], 'rationales': []} for node_type in ast_node_types + nl_pos_types} for pos, token in enumerate(experiment_result['goal_token'].tolist())}
            sample_results = {str(token_pos)+'['+str(experiment_result['goal_token'][token_pos])+']' : {node_type : {'values': [], 'rationales': []} for node_type in ast_node_types + nl_pos_types} for token_pos in range(1, len(experiment_result['rational_pos']))}
            for target_idx, target_token in enumerate(experiment_result['goal_token'].tolist()): 
                if target_idx > 0: # INITIAL TOKEN IS IGNORED
                    for rational_idx, rational_pos in enumerate(experiment_result['rationale_pos_tgt'][target_idx]):
                        for rational_tag in experiment_result['tags'][rational_pos]: 
                            if rational_tag:
                                try:
                                    sample_results[str(target_idx)+'['+str(target_token)+']'][rational_tag]['values'].append(experiment_result['rationale_prob_tgt'][target_idx][rational_idx])
                                    sample_results[str(target_idx)+'['+str(target_token)+']'][rational_tag]['rationales'].append(str(rational_pos)+'['+str(experiment_result['goal_token'][rational_pos])+']')
                                except Exception as e:
                                    print('An Error Occurred')
            aggregation_results[experiment_result['from_seq_id'].unique()[0]] = clean_results(sample_results)
    return aggregation_results

# %% [markdown]
# ### Data processing

# %%
def merge_experiments(tagged_results):
    results = {exp_id: None for exp_id in tagged_results.keys()}
    for exp_id in tagged_results.keys():
        results[exp_id] = pd.concat(tagged_results[exp_id], ignore_index=True)
    return results
    

# %% [markdown]
# ## Running Experiment

# %% [markdown]
# ### Parsing

# %%
### Define parser
parser, node_types = create_parser('python')
### Defines pos tags 
pos_types = list(nltk.data.load('help/tagsets/upenn_tagset.pickle'))

# %% [markdown]
# ### Model, Tokenizer Loading

# %%
model = AutoModelForCausalLM.from_pretrained(params['model_name'], cache_dir=params['cache_dir'])
tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
model.to(device)
model.eval()

# %% [markdown]
# ### Execute

# %%
### GET RATIONALES
experiment_paths = [get_experiment_path(params['num_samples'], params['size_samples'], exp) for exp in range(params['num_experiments'])]
rationales_results_dfs = read_rationales(experiment_paths)

# %%
a = rationales_results_dfs[0]
b = rationales_results_dfs[1]

# %%
results = pd.concat([a,b], ignore_index=True)

# %%
###TAG EXPERIMENTS RESULTS - TAKES TIME
nl_ast_types = ['comment','identifier','string']
tagged_results = tag_rationals(rationales_results_dfs, nl_ast_types, pos_types, params['delimiter_sequence'], parser)

# %%
tagged_results = merge_experiments(tagged_results)

# %% [markdown]
# ### Storing the results

# %%
for exp_id, exp_result in tagged_results.items():
    exp_result.to_csv(params['tagged_rationales'] + '/' + params['dataset'] +'_exp_' + str(exp_id) +'.csv', index=False)


