# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_taxonomies.ipynb.

# %% auto 0
__all__ = ['pl_taxonomy_python', 'nl_pos_taxonomy', 'global_groups']

# %% ../nbs/00_taxonomies.ipynb 2
#Programming Language Taxonomy
def pl_taxonomy_python() -> dict:
    return {
  "punctuation": ['{', '}', '[', ']', '(', ')','\"', ',', '.', '...', ';', ':'], #NO SEMANTIC
  "exceptions": ['raise_statement','catch', 'try', 'finally', 'throw', 'throws', 'except'], #SEMANTIC
  "oop": ['def','class','instanceof','interface','private','protected','public','abstract','extends','package','this','implements','import','new','super'], #SEMANTIC
  "asserts": ['assert'], #SEMANTIC
  "types": ['tuple','set','list','pair','subscript','type','none','dictionary','integer','native','static','synchronized','transient','volatile','void','final','enum','byte','char','float','boolean','double','int','long','short','strictfp'], #SEMANTIC
  "conditionals": ['else', 'if', 'switch', 'case', 'default'], #SEMANTIC
  "loops": ['break', 'do', 'for', 'while', 'continue'], #SEMANTIC
  "operators": ['as','yield','is','@','in','and','or','not','**','slice','%','+','<','>','=','+','-','*','/','%','++','--','!','==','!=','>=','<=','&&','||','?',':','~','<<','>>','>>>','&','^','|','//'],#NO SEMANTIC
  "indentation": ['\n','\t', 'identation'],#NO SEMANTIC
  "bool": ['true', 'false'], #SEMANTIC
  "functional":['lambda','lambda_parameters'],#NO SEMANTIC
  "with" : ['with','with_item','with_statement','with_clause'], #SEMANTIC
  "return" :['return'],  #NO SEMANTIC
  "structural" : ['attribute', 'argument_list','parenthesized_expression','pattern_list','class_definition','function_definition','block'], #SEMANTIC
  "statements" : ['return_statement','break_statement','assignment','while_statement','expression_statement','assert_statement','for_statement'],#SEMANTIC
  "expression": ['call','exec','async','ellipsis','unary_operator','binary_operator','as_pattern_target','boolean_operator','as_pattern','comparison_operator','conditional_expression','named_expression','not_operator','primary_expression','as_pattern'], #NO SEMANTIC
  "errors": ["ERROR"], #ERROR
  "identifier":["identifier"],  #NL
  "comment":["comment"], #NL
  "string": ['string','interpolation','string_content','string_end','string_start','escape_sequence'], #NL
  "excluded": ['module'], ### EXCLUDED CATEGORY
  "unknown": []
}

# %% ../nbs/00_taxonomies.ipynb 3
def nl_pos_taxonomy() -> dict: return {
    "nl_verb" : ['VBN', 'VBG', 'VBZ', 'VBP', 'VBD', 'VB'],
    "nl_noun" : ['NN', 'NNPS', 'NNS', 'NNP'],
    "nl_pronoun" : ['WP', 'PRP', 'PRP$', 'WP','WP$'], 
    "nl_adverb" : ['RBS','RBR', 'RB', 'WRB'], 
    "nl_adjetive" : ['JJR', 'JJS', 'JJ'], 
    "nl_determiner" : ['DT','WDT','PDT'], 
    "nl_preposition" : ['IN', 'TO'],
    "nl_particle" : ['RP'],
    "nl_modal" : ['MD'],
    "nl_conjunction" : ['CC'],
    "nl_cardinal" : ['CD'],
    "nl_other" : ['FW', 'EX', 'SYM' , 'UH', 'POS', "''", '--',':', '(', ')', '.', ',', '``', '$', 'LS']
}

# %% ../nbs/00_taxonomies.ipynb 4
def global_groups() -> dict:
    return {
        'sc_semantic': ['exceptions', 'oop', 'asserts', 'types', 'conditionals', 'loops', 'bool', 'structural', 'statements', 'with'], 
        'sc_nl': ['identifier', 'comment', 'string'],
        'sc_not_semantic': ['punctuation', 'operators', 'indentation', 'functional', 'return', 'expression'], 
        'sc_errors' : ['errors'], 
        'nl_semantic': ['nl_verb', 'nl_noun', 'nl_pronoun', 'nl_adjetive'],
        'nl_not_semantic' : ['nl_adverb', 'nl_determiner', 'nl_preposition', 'nl_particle', 'nl_modal', 'nl_conjunction', 'nl_cardinal', 'nl_list'], 
        'unknown': ['unknown', 'nl_other'], 
        'excluded': ['excluded'], ### EXCLUDED CATEGORIES
    }
