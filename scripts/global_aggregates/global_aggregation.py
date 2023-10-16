# %% [markdown]
# # Mapping Function
# > Labels a given token following a tailored taxonomy

# %%
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import random
import numpy as np
import re

# %%
from code_rationales.loader import download_grammars
from tree_sitter import Language, Parser
import code_rationales

# %%
import nltk

# %%
from code_rationales.taxonomies import *

# %% [markdown]
# ## Setup

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# %%
def param_default():
    return {
        #'dataset' : 'code_completion_random_cut_5k_30_512_tokens',
        #'dataset' : 'code_completion_docstring_random_cut_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_signature_3.8k_30_150_tokens',
        'dataset' : 'code_completion_docstring_5k_30_150_tokens',
        'rational_results': '/workspaces/code-rationales/data/rationales/gpt',
        'global_ast_results': '/workspaces/code-rationales/data/global_ast_results/gpt',
        'global_taxonomy_results': '/workspaces/code-rationales/data/global_taxonomy_results/gpt',
        'delimiter_sequence': '',
        'num_samples' : 100, 
        'size_samples' : 157,
        'num_experiments': 30, 
        'bootstrapping' : 500
    }
params = param_default()

# %% [markdown]
# ## AST Mapping

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
            (node_span[0]-1 <= tok_span[0] and tok_span[1] <= node_span[1] and node_text in token) or \
            (tok_span[0] <= node_span[0] and node_span[1] <= tok_span[1])

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
    lines: list,     # list of lines in the code, 
    nodes_information: dict # dict with augmented information of each ast node
) -> list: 
    """Get all AST types for the given token span"""
    results = []
    for node_id, node_info in nodes_information.items():
        if is_token_span_in_node_span(tok_span, token, node_info['span'], node_info['node'].text.decode('utf-8')):
            results.append(node_info['node'])   
    return results

# %%
def get_node_height(node):
    if not node.children: 
        return 0
    children_heights = []
    for child in node.children:
        children_heights.append(get_node_height(child))
    return max(children_heights) + 1

# %%
def augment_ast(node, lines):
    """Get an array with additional infor for each node in the AST, Appends the height and span"""
    information = {}
    def traverse_and_append_info(node, lines, information):
        information[node.id] = {'height': get_node_height(node), 'span': get_node_span(node, lines), 'node': node}
        for child in node.children:
            traverse_and_append_info(child, lines, information)
    traverse_and_append_info(node, lines, information)
    return information 

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
# ### Identation Mappings

# %%
def get_identation_spans(source_code:str):
    pattern = '\s+(?=\w)|\t|\n'
    return [(m.start(0), m.end(0)-1) for m in re.finditer(pattern, source_code)]

# %% [markdown]
# ## Taxonomy Mapping 

# %%
def clean_results(global_results):
    def clean_dictonary(result_dict):
        clean_dict = result_dict.copy()
        for key, value in result_dict.items():
            if not value: 
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
def map_to_taxonomy(sc_taxonomy_dict:dict, nl_taxonomy_dict: dict, result_dict: dict):
    result_dict = result_dict.copy()
    mappings = {token: {category : [] for category in {**sc_taxonomy_dict, **nl_taxonomy_dict}.keys()} for token in {**sc_taxonomy_dict, **nl_taxonomy_dict}.keys()}
    for target_token, value in result_dict.items():
        for source_token, rationales_values in value.items():
            try: 
                if source_token[:2] == 'sc':
                    mappings[search_category_by_token(sc_taxonomy_dict, target_token.split('_|_')[1])][search_category_by_token(sc_taxonomy_dict, source_token.split('_|_')[1])] += rationales_values 
                elif source_token[:2] == 'nl':
                    mappings[search_category_by_token(sc_taxonomy_dict, target_token.split('_|_')[1])][search_category_by_token(nl_taxonomy_dict, source_token.split('_|_')[1])] += rationales_values
            except Exception as e:
                print(e)
    return clean_results(mappings)

# %%
def map_global_results_to_taxonomy(sc_taxonomy_dict:dict, nl_taxonomy_dict:dict, results: dict):
    return dict(zip(results.keys(), map(lambda aggegrations: map_to_taxonomy(sc_taxonomy_dict, nl_taxonomy_dict, aggegrations), results.values())))

# %% [markdown]
# ##  Rationals Tagging

# %%
calculate_right_span = lambda start_idx, end_idx, df : len(''.join(map(str, df.loc[start_idx:end_idx, 'goal_token'].tolist())))
calculate_span = lambda right_span, token : (right_span-len(str(token)), right_span)
delete_leading_spaces = lambda string: re.sub(r'^\s+', '', string)
delete_leading_breaks = lambda string: re.sub(r'^\n+', '', string)

# %%
def add_first_token_row(df):
    df.loc[-1] = [df['typesets_tgt'][0][0][0], df['from_seq_id'][0], None, None, None]
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
        if nl_tags: df.at[idx, 'tags'] = df['tags'][idx] + [('nl',nl_tags[-1],0)]
    ##### POS TAGS FOR CODE PART
    target_code = ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    nl_target_nodes = get_nodes_by_type(parser.parse(bytes(target_code, 'utf8')).root_node, nl_ast_types)
    for token_idx in range(df[df['token_type'] == 'src'].first_valid_index(), len(df['span'])):
                for nl_target_node in nl_target_nodes:
                    if is_token_span_in_node_span(df['span'][token_idx], df['goal_token'][token_idx], get_node_span(nl_target_node, target_code.split("\n")), nl_target_node.text.decode('utf-8')) and \
                            (str(df['goal_token'][token_idx]) in nl_target_node.text.decode('utf-8') or nl_target_node.text.decode('utf-8') in str(df['goal_token'][token_idx])):
                            tagged_token_list = list(filter(lambda tagged_token: str(tagged_token[0]).replace(' ','') in str(df['goal_token'][token_idx]).replace(' ','') or str(df['goal_token'][token_idx]).replace(' ','') in str(tagged_token[0]).replace(' ',''), \
                                                        nltk.pos_tag( nltk.word_tokenize(nl_target_node.text.decode('utf-8')))))
                            if len(tagged_token_list)>0 and tagged_token_list[0][1] in nl_pos_types and tagged_token_list[0][1] not in df['tags'][token_idx]: 
                                    df.at[token_idx, 'tags'] = df['tags'][token_idx] + [('nl', tagged_token_list[0][1],0)]

# %%
def fill_ast_tags_in_experiment_result(df, parser):
    target_code = ''.join(df[df['token_type'] == 'src']['goal_token'].map(lambda value: str(value)))
    src_initial_token_idx = df[df['token_type'] == 'src'].first_valid_index()
    target_ast = parser.parse(bytes(target_code, 'utf8')).root_node
    nodes_information = augment_ast(target_ast, target_code.split("\n"))
    identation_spans = get_identation_spans(target_code)
    for token_idx in range(src_initial_token_idx, len(df)):
        df.at[token_idx, 'tags'] = df['tags'][token_idx] + list(map(lambda node: ('sc', node.type, nodes_information[node.id]['height']), 
                                                                    get_token_nodes(df['span'][token_idx], df['goal_token'][token_idx], target_code.split("\n"), nodes_information)))
        df.at[token_idx, 'tags'] = df['tags'][token_idx] + list(map(lambda iden_span: ('sc','identation', 0), filter(lambda iden_span: is_token_span_in_node_span(df['span'][token_idx],  df['goal_token'][token_idx], iden_span, target_code[iden_span[0]:iden_span[1]+1]), identation_spans)))

# %%
def tag_rationals(experiment_paths: list, nl_ast_types: list, nl_pos_types: list, delimiter_sequence: str, parser):
    experiments = {}
    for exp_idx, experiment_path in enumerate(experiment_paths):
        experiment_results = []
        df_experiment = pd.read_csv(experiment_path, index_col=0)
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
# ## Rationals Aggregation

# %%
def aggregate_rationals(global_tagged_results: dict, ast_node_types: list, nl_pos_types: list):
    aggregation_results = {exp_id: None for exp_id in global_tagged_results.keys()}
    for exp_idx, experiment_results in global_tagged_results.items():
        experiment_aggregation_results = {node_type: {**{ 'sc'+'_|_'+node_type : [] for node_type in ast_node_types }, **{ 'nl'+'_|_'+node_type: [] for node_type in nl_pos_types }} 
                                          for node_type in {**{ 'sc'+'_|_'+node_type : {'values': [], 'rationales': []} for node_type in ast_node_types }, **{ 'nl'+'_|_'+node_type : {'values': [], 'rationales': []} for node_type in nl_pos_types }}}
        print('*'*10 +'Aggregrating rationals for exp: ' +str(exp_idx) + '*'*10)
        for result_idx, experiment_result in enumerate(experiment_results):
            for target_idx in range(len(experiment_result)):
                if target_idx > 0: # INITIAL TOKEN IS IGNORED
                    for rational_idx, rational_pos in enumerate(eval(experiment_result['rationale_pos_tgt'][target_idx])):
                        try:
                            [experiment_aggregation_results[target_tag[0]+'_|_'+target_tag[1]][rational_tag[0]+'_|_'+rational_tag[1]].append(eval(experiment_result['rationale_prob_tgt'][target_idx])[rational_idx]) if target_tag[1] and rational_tag[1] else None \
                            for rational_tag in experiment_result['tags'][rational_pos] for target_tag in experiment_result['tags'][target_idx]]
                        except Exception as e:
                            print('An Error Occurred')
        aggregation_results[exp_idx] = clean_results(experiment_aggregation_results)
    return aggregation_results

# %% [markdown]
# ## Bootstrapping

# %%
def bootstrapping( np_data, np_func, size ):
    """Create a bootstrap sample given data and a function
    For instance, a bootstrap sample of means, or mediands. 
    The bootstrap replicates are a long as the original size
    we can choose any observation more than once (resampling with replacement:np.random.choice)
    """
    
    #Cleaning NaNs
    #np_data_clean = np_data[ np.logical_not( np.isnan(np_data) ) ] 
    
    #The size of the bootstrap replicate is as big as size
    #Creating the boostrap replicates as long as the orignal data size
    #This strategy might work as imputation 
    bootstrap_repl = [ np_func( np.random.choice( np_data, size=len(np_data) ) ) for i in range( size ) ]
    
    #logging.info("Covariate: " + cov) #Empirical Mean
    #logging.info("Empirical Mean: " + str(np.mean(np_data_clean))) #Empirical Mean
    #logging.info("Bootstrapped Mean: " + str( np.mean(bootstrap_repl) ) ) #Bootstrapped Mean
    
    return np.array( bootstrap_repl )

# %%
def bootstrap_samples_global_results(global_results: dict, size: int):
    for exp_id in global_results.keys():
        experiment_result = global_results[exp_id]
        for target_type, target_value in global_results[exp_id].items():
            for source_type, source_value in target_value.items():
                experiment_result[target_type][source_type] = bootstrapping(source_value, np.mean, size).tolist()
        global_results[exp_id] = experiment_result

# %% [markdown]
# ## Running Experiment

# %%
### Retrieve experiments
get_experiment_path =  lambda samples, size, exp: params['rational_results'] + '/' + params['dataset'] + '/' + '[t_'+str(samples)+']_[max_tgt_'+str(size)+']_[exp:'+str(exp)+'].csv'
experiment_paths = [get_experiment_path(params['num_samples'], params['size_samples'], exp) for exp in range(params['num_experiments'])]
### Define parser
parser, node_types = create_parser('python')
node_types += ['identation']
### Defines pos tags 
pos_types = list(nltk.data.load('help/tagsets/upenn_tagset.pickle'))

# %%
###TAG EXPERIMENTS RESULTS - TAKES TIME
nl_ast_types = ['comment','identifier','string']
global_tagged_results = tag_rationals(experiment_paths, nl_ast_types, pos_types, params['delimiter_sequence'], parser)

# %%
###AGGREGATE RATIONALS - TAKES TIME
global_aggregated_results = aggregate_rationals(global_tagged_results, node_types, pos_types)

# %%
###GROUP AGGREGATES BY TAXONOMY
global_taxonomy_results = map_global_results_to_taxonomy(pl_taxonomy_python(), nl_pos_taxonomy(), global_aggregated_results.copy())

# %% [markdown]
# ## Storing Results

# %%
for exp_id, exp_aggregation in global_aggregated_results.items():
    with open(params['global_ast_results'] + '/' + params['dataset'] +'_exp_' + str(exp_id) +'.txt', 'w') as file:
        file.write(json.dumps(exp_aggregation))

# %%
for exp_id, exp_aggregation in global_taxonomy_results.items():
    with open(params['global_taxonomy_results'] + '/' + params['dataset'] +'_exp_' + str(exp_id) +'.txt', 'w') as file:
        file.write(json.dumps(exp_aggregation))


