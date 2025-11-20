
from flask import Flask
from flask import render_template 
from flask import request, jsonify
import random
import torch
import importlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

# imports rationalization functions from sequential-rationales library
rationalization = importlib.import_module("rationalization")
rationalize_lm = rationalization.rationalize_lm

# EB looks for an 'application' callable by default.

#Note to everyone: run "npm run build" after changes to js files
application = Flask(__name__, template_folder="./src/frontend",static_folder="./src/frontend")

#given prompt is only Natural Language, added code is only Programming Language
conceptsNat = dict()
conceptsNat["Semantic"] = ["Noun", "Verb", "pronouns"]
conceptsNat["Non-semantic"] = ["Preposition", "Determiner", "Adverb", "Adjective", "Cardinal", "Particle", "Model", "Conjunction", "List"]

conceptsProgram = dict()
conceptsProgram["Semantic"] = ["Types", "Exceptions", "OOP", "Conditional", "Loops", "Bool", "Structural", "With", "Asserts", "Statements"]
conceptsProgram["Natural Language in Code"] = ["Identifier", "Comment", "String"]
conceptsProgram["Syntax"] = ["Errors"]
conceptsProgram["Non-Semantic"] = ["Expression", "Punctuation", "Operators", "Indentation", "Functional", "Return"]
conceptsProgram["Context Window"] = ["Class", "Constructor", "Signature", "Field", "Focal Method"]

randomCode = ["class", "def", "list", "dict", "(", ")", ":", "\t", "\n", "if", "elif", "else", "while", "for", "range", "return" ]

delimiters = [",", ".", "(", ")", "{", "}", "[", "]", ":", ";", "\n", "\t"]

#add tokens from either original prompt or a list of random bits of code
def addCode(tokens):
    new = ""
    for i in range(random.randint(1,20)):
        choice = random.random()
        if choice < 0.5:
            n = random.choice(tokens)
        else:
            n = random.choice(randomCode)
        if n == "(" or n == "[" or n == "{":
            new += n
        elif n == ")" or n == "]" or n == "}":
            new = new.rstrip()
            new += n
        else:
            new += n
            new += " "
    return new

def highIndex(probList):
    highest = -1
    index = -1
    for i in range(len(probList)):
        if probList[i] > highest:
            highest = probList[i]
            index = i
    return index

def indexSort(wordList, countList, probList):
    """
    Selection sort by smallest index
    """
    for i in range(len(countList)):
        small = i
        smallVal = countList[i]
        for j in range(i+1,len(countList)):
            if countList[j] < smallVal:
                small = j
                smallVal = countList[j]
        wordTemp = wordList[i]
        countTemp = countList[i]
        probTemp = probList[i]
        wordList[i] = wordList[small]
        wordList[small] = wordTemp
        countList[i] = countList[small]
        countList[small] = countTemp
        probList[i] = probList[small]
        probList[small] = probTemp
    return wordList, countList, probList


#pick the top 4-7 rationales with the highest probabilities
def chooseRationales(wordList, countList, probList):
    #make copies of lists to search through and delete from
    words = wordList[:]
    counts = countList[:]
    probs = probList[:]
    chooseWord = []
    chooseCount = []
    chooseProb = []
    for i in range(random.randint(4,7)):
        #print(words)
        high = highIndex(probs)
        chooseWord.append(words[high])
        chooseCount.append(counts[high])
        chooseProb.append(probs[high])

        del words[high]
        del counts[high]
        del probs[high]
        if len(words) == 0:
            return indexSort(chooseWord, chooseCount, chooseProb)
    #need to fix them being out of order
    return indexSort(chooseWord, chooseCount, chooseProb)

def splitUpTokens(prompt):
    wordsList = prompt.split(" ")

    wordIndex = 0
    while wordIndex < len(wordsList):
        for delimiter in delimiters:
            if delimiter in wordsList[wordIndex]:
                preslice = wordsList[0:wordIndex]
                postslice = wordsList[wordIndex + 1:]

                delimiterIndex = wordsList[wordIndex].find(delimiter)
                slicedWord = []
                if delimiterIndex != 0:  # don't include empty strings
                    slicedWord.append(wordsList[wordIndex][0:delimiterIndex])
                slicedWord.append(delimiter)  
                if len(wordsList[wordIndex][delimiterIndex + 1:]) != 0:  # don't include empty strings
                    slicedWord.append(wordsList[wordIndex][delimiterIndex + 1:])
                
                wordsList = preslice + slicedWord + postslice
        
        wordIndex += 1

    return wordsList

def makeRationales(prompt):
    """
    Uses the rationalization model and returns a rationales json to be returned to frontend
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'model')

    model = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    outputs = model.generate(input_ids=input_ids, max_length=25, do_sample=False)[0]

    torch.cuda.empty_cache() #Cleaning Cache
    all_rationales, log = rationalize_lm(model, outputs, tokenizer, verbose=True)

    rationales = parse_data(log, prompt)
    return rationales

def parse_data(input, prompt):
    """
    Formats the model output data to match frontend json format
    """
    rationales = dict()
    rationales["_phrase"] = ''.join(input['input_text'])

    # Blank data for the first rationale index
    rationales['0'] = {
        "token": input["input_text"][0],
        "rationales": [],
        "rationales_indexes": [],
        "probabilities": [],
        "concept_view": ["Natural Language", "Semantic", "Noun"]
    }

    for obj in input["rationalization"]:
        data = {}
        data["token"] = obj["goal_word"]
        data["rationales"] = []
        data["rationales_indexes"] = []
        data["probabilities"] = []

    # randomly choosing between concept tree or programming tree taxonomy
        useNatural = random.choice([0,1])
        if useNatural:
            middle = random.choice(list(conceptsNat.keys()))
            leaf = random.choice(conceptsNat[middle])
            data["concept_view"] = ["Natural Language", middle, leaf]
        else:
            middle = random.choice(list(conceptsProgram.keys()))
            leaf = random.choice(conceptsProgram[middle])
            data["concept_view"] = ["Programming Language", middle, leaf]


        for item in obj["log"]:
            data["rationales"].append( item["added_token_text"] )
            data["rationales_indexes"].append( item["added_token_position"] )
            data["probabilities"].append( item["true_token_prob"] )

        # Sorting the rationales given from the model
        data["rationales"], data["rationales_indexes"], data["probabilities"] = indexSort(
            data["rationales"],
            data["rationales_indexes"],
            data["probabilities"]
        )

        rationales[ str(obj["target_position"]) ] = data
    
    return rationales

@application.route("/") 
def hello():            
    message = "Hello, World"        
    return render_template('index.html',  
                           message=message) 
          
@application.route("/prompt", methods = ['POST', 'GET'])
def parsePrompt():
    content = request.json
    prompt = content['prompt']
    return jsonify(makeRationales(prompt))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()