# %% [markdown]
# # Mapping Function
# > Labels a given token following a tailored taxonomy

# %%
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import random
import numpy as np

# %%
from code_rationales.loader import download_grammars
from tree_sitter import Language, Parser
import code_rationales

# %% [markdown]
# ## Java Taxonomy

# %%
#Programming Language Taxonomy
def pl_taxonomy_java() -> dict:
    return {
  "parenthesis": { #Category-Level Label
    "<{>": "{", #Token-Level Label
    "<}>": "}",
    "<[>": "[",
    "<]>": "]",
    "<(>": "(",
    "<)>": ")"    
  },
  "semi_colon":{
    "<;>": ";",
    "<:>": ":"
  },
  "comma_dot":{
    "<,>": ",",
    "<.>": ".",
    "<...>": "..."
  },
  "exceptions": {
    "<catch>": "catch",
    "<try>": "try",
    "<finally>": "finally",
    "<throw>": "throw",
    "<throws>": "throws"
  },
  "oop": {
    "<class>": "class",
    "<instanceof>": "instanceof",
    "<interface>": "interface",
    "<private>": "private",
    "<protected>": "protected",
    "<public>": "public",
    "<abstract>": "abstract",
    "<extends>": "extends",
    "<package>": "package",
    "<this>": "this",
    "<implements>": "implements",
    "<import>": "import",
    "<new>": "new",
    "<super>": "super"
  },
  "asserts": {
    "<assert>": "assert"
  },
  "types": {
    "<native>": "native",
    "<static>": "static",
    "<synchronized>": "synchronized",
    "<transient>": "transient",
    "<volatile>": "volatile",
    "<void>": "void",
    "<final>": "final",
    "<enum>": "enum",
    "<byte>": "byte",
    "<char>": "char",
    "<float>": "float",
    "<boolean>": "boolean",
    "<double>": "double",
    "<int>": "int",
    "<long>": "long",
    "<short>": "short",
    "<strictfp>": "strictfp"
  },
  "conditionals": {
    "<else>": "else",
    "<if>": "if",
    "<switch>": "switch",
    "<case>": "case",
    "<default>": "default"
  },
  "loops": {
    "<break>": "break",
    "<do>": "do",
    "<for>": "for",
    "<while>": "while",
    "<continue>": "continue"
  },
  "operators": {
    "<=>": "=",
    "<+>": "+",
    "<->": "-",
    "<*>": "*",
    "</>": "/",
    "<%>": "%",
    "<++>": "++",
    "<-->": "--",
    "<!>": "!",
    "<==>": "==",
    "<!=>": "!=",
    "<greater_equal>": ">=",
    "<lesser_equal>": "<=",
    "<&&>": "&&",
    "<||>": "||",
    "<?>": "?",
    "<:>": ":",
    "<~>": "~",
    "<double_lesser>": "<<",
    "<double_greater>": ">>",
    "<triple_greater>": ">>>",
    "<&>": "&",
    "<^>": "^",
    "<|>": "|"
  },
  "newline": {
    "<n>": "\n"
  },
  "tab": {
    "<t>": "\t"
  },
  "ampersand": {
    "<@>": "@"
  },
  "bool": {
    "<true>": "true",
    "<false>": "false",
  }
}

# %%
def map_from_preprocessed_java( preprocessed_sequence ) -> list:
    a = preprocessed_sequence
    return list(a)

# %% [markdown]
# ## Python Taxonomy

# %%
#Programming Language Taxonomy
def pl_taxonomy_python() -> dict:
    return {
  "parenthesis": { #Category-Level Label
    "<{>": "{", #Token-Level Label
    "<}>": "}",
    "<[>": "[",
    "<]>": "]",
    "<(>": "(",
    "<)>": ")"    
  },
  "semi_colon":{
    "<;>": ";",
    "<:>": ":"
  },
  "comma_dot":{
    "<,>": ",",
    "<.>": ".",
    "<...>": "..."
  },
  "exceptions": {
    "<catch>": "catch",
    "<try>": "try",
    "<finally>": "finally",
    "<throw>": "throw",
    "<throws>": "throws"
  },
  "oop": {
    "<class>": "class",
    "<instanceof>": "instanceof",
    "<interface>": "interface",
    "<private>": "private",
    "<protected>": "protected",
    "<public>": "public",
    "<abstract>": "abstract",
    "<extends>": "extends",
    "<package>": "package",
    "<this>": "this",
    "<implements>": "implements",
    "<import>": "import",
    "<new>": "new",
    "<super>": "super"
  },
  "asserts": {
    "<assert>": "assert"
  },
  "types": {
    "<native>": "native",
    "<static>": "static",
    "<synchronized>": "synchronized",
    "<transient>": "transient",
    "<volatile>": "volatile",
    "<void>": "void",
    "<final>": "final",
    "<enum>": "enum",
    "<byte>": "byte",
    "<char>": "char",
    "<float>": "float",
    "<boolean>": "boolean",
    "<double>": "double",
    "<int>": "int",
    "<long>": "long",
    "<short>": "short",
    "<strictfp>": "strictfp"
  },
  "conditionals": {
    "<else>": "else",
    "<if>": "if",
    "<switch>": "switch",
    "<case>": "case",
    "<default>": "default"
  },
  "loops": {
    "<break>": "break",
    "<do>": "do",
    "<for>": "for",
    "<while>": "while",
    "<continue>": "continue"
  },
  "operators": {
    "<=>": "=",
    "<+>": "+",
    "<->": "-",
    "<*>": "*",
    "</>": "/",
    "<%>": "%",
    "<++>": "++",
    "<-->": "--",
    "<!>": "!",
    "<==>": "==",
    "<!=>": "!=",
    "<greater_equal>": ">=",
    "<lesser_equal>": "<=",
    "<&&>": "&&",
    "<||>": "||",
    "<?>": "?",
    "<:>": ":",
    "<~>": "~",
    "<double_lesser>": "<<",
    "<double_greater>": ">>",
    "<triple_greater>": ">>>",
    "<&>": "&",
    "<^>": "^",
    "<|>": "|"
  },
  "newline": {
    "<n>": "\n"
  },
  "tab": {
    "<t>": "\t"
  },
  "ampersand": {
    "<@>": "@"
  },
  "bool": {
    "<true>": "true",
    "<false>": "false",
  }
}

# %% [markdown]
# ## AST Mapping

# %% [markdown]
# ### Calculate Spans

# %%
#df_rationals = pd.read_csv('/workspaces/code-rationales/data/rationales/gpt/testing/[t_100]_[max_tgt_44]_[exp:0]_.csv',index_col=0)

# %%
#df_rationals = df_rationals[df_rationals['from_seq_id'] == 0]

# %%
### Retrieve the generated output
#initial_token = eval(df_rationals['typesets_tgt'][0])[0][0]
#code = initial_token + ''.join(df_rationals['goal_token'])
#code

# %%
#### Add Span column
#calculate_left_span = lambda index : len(initial_token + ''.join(df_rationals['goal_token'][:index]))
#calculate_right_span = lambda left_span, token : len(left_span) + len(token)
#span_col = list(map(lambda tuple: (tuple[0],tuple[0]+len(tuple[1])),[(calculate_left_span(index),token) for index, token in df_rationals['goal_token'].items()]))
#df_rationals.insert(loc=df_rationals.columns.get_loc('goal_token')+1, column='span', value=span_col)

# %%
#df_rationals

# %% [markdown]
# ### Map Tokens with Nodes

# %%
languages=['python', 'java']
download_grammars(languages)

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
def get_token_type(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    nodes: list,     # list of tree-sitter nodes
    lines: list,     # list of lines in the code
) -> tuple: # (parent_type, token_type) of the token
    """Get the parent AST type and token AST type of a token."""
    node_spans = [get_node_span(node, lines) for node in nodes]
    for i, span in enumerate(node_spans):
        if (span[0] <= tok_span[0] and tok_span[0] < span[1]) or (span[0] < tok_span[1] and tok_span[1] <= span[1]):
            return nodes[i].parent.type, nodes[i].type

# %%
def get_token_nodes(
    tok_span: tuple, # (start, end) position of a token in tokenizer
    node,            # tree-sitter node
    lines: list,     # list of lines in the code
) -> list: 
    """Get all AST types for the given token span"""
    results = []
    def traverse_and_get_types(tok_span, node, lines, results) -> None:
        node_span = get_node_span(node, lines)
        if (node_span[0] <= tok_span[0] and tok_span[0] < node_span[1]) or (node_span[0] < tok_span[1] and tok_span[1] <= node_span[1]):
            results.append(node.type)
        for n in node.children:
            traverse_and_get_types(tok_span, n, lines, results)
    traverse_and_get_types(tok_span, node, lines, results)
    return results

# %%
#parser, node_types = create_parser('python')

# %%
#nodes = traverse(parser.parse(bytes(code, 'utf8')).root_node)

# %%
#print(get_token_type(df_rationals['span'][40], nodes, code.split("\n")))

# %%
#print(get_token_nodes(df_rationals['span'][42], parser.parse(bytes(code, 'utf8')).root_node, code.split("\n")))

# %%
#print(eval(df_rationals['rationale_pos_tgt'][2]))
#print(eval(df_rationals['rationale_prob_tgt'][2]))

# %%
#print(df_rationals['goal_token'][eval(df_rationals['rationale_pos_tgt'][2])[0]-1])
#print(df_rationals['span'][eval(df_rationals['rationale_pos_tgt'][2])[0]-1])


# %% [markdown]
# ##  Rational Global Aggregates

# %%
def param_default():
    return {
        #'dataset' : 'code_completion_random_cut_5k_30_512_tokens',
        'dataset' : 'code_completion_docstring_random_cut_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_signature_3.8k_30_150_tokens',
        #'dataset' : 'code_completion_docstring_5k_30_150_tokens',
        'rational_results': '/workspaces/code-rationales/data/rationales/gpt',
        'global_results': '/workspaces/code-rationales/data/global_results/gpt',
        'num_samples' : 100, 
        'size_samples' : 146,
        'num_experiments': 30, 
        'bootstrapping' : 500
    }
params = param_default()

# %%
get_experiment_path =  lambda samples, size, exp: params['rational_results'] + '/' + params['dataset'] + '/' + '[t_'+str(samples)+']_[max_tgt_'+str(size)+']_[exp:'+str(exp)+']_.csv'
calculate_left_span = lambda index, initial_token, df_rationals : len(initial_token + ''.join(df_rationals['goal_token'][:index]))
calculate_right_span = lambda left_span, token : len(left_span) + len(token)

# %%
### Retrieve experiments
experiment_paths = [get_experiment_path(params['num_samples'], params['size_samples'], exp) for exp in range(params['num_experiments'])]
### Define parser
parser, node_types = create_parser('python')

# %%
def aggregate_rationals(experiment_paths: list, parser, node_types: list):
    global_results = {node_type : {node_type : [] for node_type in node_types} for node_type in node_types}
    for exp_idx, experiment_path in enumerate(experiment_paths):
        df_experiment = pd.read_csv(experiment_path, index_col=0)
        experiment_rational_results = [df_experiment[(df_experiment['from_seq_id'] == sample_idx) | (df_experiment['from_seq_id'] == str(sample_idx))].reset_index() for sample_idx in range(params['num_samples'])]
        print('*'*10 +'Aggregating rationales for exp: ' +str(exp_idx) + '*'*10)
        for experiment_rational_result in experiment_rational_results:
            initial_token = eval(experiment_rational_result['typesets_tgt'][0])[0][0]
            experiment_rational_result.insert(loc=experiment_rational_result.columns.get_loc('goal_token')+1, column='span', value=list(map(lambda tuple: (tuple[0],tuple[0]+len(tuple[1])),[(calculate_left_span(index, initial_token, experiment_rational_result), str(token)) for index, token in experiment_rational_result['goal_token'].items()])))
            target_code = eval(experiment_rational_result['typesets_tgt'][0])[0][0] + ''.join(str(experiment_rational_result['goal_token']))
            target_ast = parser.parse(bytes(target_code, 'utf8')).root_node
            for target_token_idx in range(len(experiment_rational_result['span'])):
                target_node_types = get_token_nodes(experiment_rational_result['span'][target_token_idx], target_ast, target_code.split("\n"))
                for rational_idx, rational_pos in enumerate(eval(experiment_rational_result['rationale_pos_tgt'][target_token_idx])):
                    if eval(experiment_rational_result['rationale_pos_tgt'][target_token_idx])[rational_idx] > 0: #rational 1 position.
                        try:
                            rational_node_types = get_token_nodes(experiment_rational_result['span'][eval(experiment_rational_result['rationale_pos_tgt'][target_token_idx])[rational_idx]-1], target_ast, target_code.split("\n"))
                            [global_results[target_node_type][rational_node_type].append(eval(experiment_rational_result['rationale_prob_tgt'][target_token_idx])[rational_idx]) for target_node_type in target_node_types for rational_node_type in rational_node_types]
                        except Exception as e:
                            print('rational pos out of range')
    return global_results

# %%
def clean_global_results(global_results):
    def clean_dictonary(result_dict):
        clean_dict = result_dict.copy()
        for key, value in result_dict.items():
            if not value:
                clean_dict.pop(key)
        return clean_dict
    for key, value in global_results.items():
        global_results[key] = clean_dictonary(value)
    return clean_dictonary(global_results)

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
    for target_type, target_value in global_results.items():
        for source_type, source_value in target_value.items():
            global_results[target_type][source_type] = bootstrapping(source_value, np.mean, size).tolist()

# %%
### WARNING TAKES TIME
global_results = clean_global_results(aggregate_rationals(experiment_paths, parser, node_types))

# %%
### WARNING TAKES TIME
bootstrap_samples_global_results(global_results, params['bootstrapping'])

# %%
with open(params['global_results'] + '/' + params['dataset'] + '.txt', 'w') as file:
    file.write(json.dumps(global_results))

# %%
with open(params['global_results'] + '/' + params['dataset'] + '.txt', 'r') as file:
    global_results = json.load(file)
print(global_results['identifier']['identifier'])


