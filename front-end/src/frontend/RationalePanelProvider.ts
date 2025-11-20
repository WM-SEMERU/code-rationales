/* Creates the Rationale panel at the top of the extension.*/ 

import * as vscode from 'vscode';
import { flaskServerRunning , flaskServerReady} from '../extension';
import MaxHeap from 'heap-js';
import { VisualizationPanelProvider } from './VisualizationPanelProvider';
import { json } from 'node:stream/consumers';

const example1 = require('../backend/example1.json');
const example2 = require('../backend/example2.json');
const example3 = require('../backend/example3.json');

/** Stores the rationale data. View the example.json files for practical examples.*/
interface RationaleData {
  token: string;
  rationales: string[];
  rationales_indexes: number[];
  probabilities: number[];
  concept_view: string[];
}

interface rationaleDataStructure {
  _phrase?: string;
  [key: number]: RationaleData | undefined;
}

export class RationalePanelProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  rationaleJsonFile: rationaleDataStructure;
  visualizationPanel: VisualizationPanelProvider;

  private _max_input_length: number;


  constructor(private readonly _extensionUri: vscode.Uri, visualizationPanel: VisualizationPanelProvider) {
    this.rationaleJsonFile = {};
    this.visualizationPanel = visualizationPanel;    this._max_input_length=150;
  }

  public async resolveWebviewView(webviewView: vscode.WebviewView, context: vscode.WebviewViewResolveContext, _token: vscode.CancellationToken) {

    this._view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri]
    };
    this.showError('Rationale Panel', 'Select some text to see the rationale.');

    webviewView.onDidChangeVisibility(async () => {
      if (webviewView.visible) {
        let newRationale = this.getUserHighlightedCode();
        if (newRationale !== ""){
          //handle input length errors
          if(newRationale.length > this._max_input_length){
            this.displayError(new Error("The input code exceeds the maximum length."));
            return;
          }else{
            this.showError('Rationale Panel',this.getUserHighlightedCode());
          }
        }

        console.log("Current rationale", newRationale);
        
        /* Handles input source */
        if(["example1","example2","example3"].includes(newRationale)){
          console.log("Example data is being used");
          switch(newRationale){
            case "example1": {
              this.rationaleJsonFile = example1;
              break;
            }
            case "example2": {
              this.rationaleJsonFile = example2;
              break;
            }
            case "example3": {
              this.rationaleJsonFile = example3;
              break;
            }

          }

        }
        else{
          await flaskServerReady;
          console.log("\x1b[0mFlask server is ready, sending user selected code (",newRationale,") to model\x1b[0m");
          
          //display loading notification for generating rationales
          let results = await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Generating code snippet.",
            cancellable: false
          }, (progress, token) => {
            return sendCodeToModel(newRationale); 
          });
          
          //handle errors from the model
          if(results instanceof Error){
            console.log('Error generating code snippet: '+results.message);
            this.displayError(results);
          }else{
            this.rationaleJsonFile = results;
            console.log("\x1b[0mGENERATED CODE: \x1b[0m", this.rationaleJsonFile);
          }
        }
        this.updateContent('Rationale Panel',this.rationaleJsonFile);
      }
      else{
        this.visualizationPanel.clearVisualizationPanel();
      }

    });


    webviewView.webview.onDidReceiveMessage(message => {
      switch (message.command) {
        case 'tokenClicked':
          this.handleTokenClick(Number(message.id));
        return;
      }
    });
  }

  public setRationaleJsonFile(jsonFile: {}){
    this.rationaleJsonFile = jsonFile;
    this.visualizationPanel.updateRationale(jsonFile);
  }

    public async updateContent(title: string, content: object) {
      this.setRationaleJsonFile(this.rationaleJsonFile);
      if (this._view) {
        if (Object.keys(this.rationaleJsonFile).length === 0){
          console.log("error, empty json file");
        }
        let message = 'Click a token to see rationale';
        const htmlContent = await this._getHtmlForWebview(this._view.webview, title, message, true);
        this._view.webview.html = htmlContent;
      }
    }

    public async showError(title: string, message: string) {
      if (this._view) {
        const htmlContent = await this._getHtmlForWebview(this._view.webview, title, message, false);
        this._view.webview.html = htmlContent;
      }
  }
    

  private getUserHighlightedCode(){
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showInformationMessage('No active editor found!');
        return "";
      }
      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);
    
      if (!selectedText) {
        vscode.window.showInformationMessage('Please select some text.');
        return "";
      }
      return selectedText;
    
  }


  private handleTokenClick(id: number){ 
    let highestIds = [];
    let token = this.rationaleJsonFile[id];
    let selectedToken = token?.token;
    const maxHeap = new MaxHeap<{ probability: number; index: number }>((a, b) => b.probability - a.probability);

    if (token && token.probabilities && token.rationales_indexes) {
      token.probabilities.forEach((value, index) => {
        maxHeap.push({ probability: value, index: token.rationales_indexes[index] });
      });
    }


    this._view?.webview.postMessage({
      command: 'changeColor',
      id: String(id),
      color: `rgba(0, 98, 114, 1)`,
      selectedToken: selectedToken
    });
    
    /* Arranges tokens based on highest influence of selected token */
    for(let i = 0; i < 4; i++){
      if (maxHeap.peek()){
        let topProbability = maxHeap.pop();
        let token = undefined;
        if (topProbability !== undefined ){
          token = this.rationaleJsonFile[topProbability.index]?.token;
          if(token === "\n"){
            token = "\\n";
          }
        }

        
        if (topProbability) {
          highestIds.push(topProbability.index);
          this._view?.webview.postMessage({
            command: 'changeColor',
            id: String(topProbability.index),
            color: `rgba(173, 76, 159,${topProbability.probability * 5})`, 
            token: token,
            probability: (topProbability.probability * 100).toFixed(2),
            selectedToken: selectedToken

          }); 
        }
      }
    }
    
    this.visualizationPanel.displayVisualization(highestIds, id);
  }

  private async _getHtmlForWebview(webview: vscode.Webview, title: string, content: string, isSuccessful: boolean): Promise<string> {

    /* Creates the HTML display in the Rationale panel */
    
    let showableContent = '';
    if (isSuccessful) {
      showableContent = isSuccessful ? Object.values(this.rationaleJsonFile).filter((item): item is RationaleData => item !== undefined && item.token !== undefined)
        .map((item, index) =>{
          if (item.token === "\n" || item.token === "\\n"){

            return `<span class="token" id="token-${index}" onclick="handleTokenClick('${index}')">&#92;n</span> <br>`;
          }
          else{
            return `<span class="token" id="token-${index}" onclick="handleTokenClick('${index}')">${item.token}</span>`;
        }}).join(''): `<p>${content}<\p>`;
    } 
    else {
      showableContent = `<p>${content}</p>`;
    }
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'src', 'frontend', 'style.css'));

    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>${title}</title>
      <link href="${styleUri}" rel="stylesheet">
      <style>
        .token {
          cursor: pointer;
          padding: 2px;
          margin: 1px;
          border: 1px solid #ccc;
          display: inline-block;
        }
      </style>
    </head>
    <body>
      <h1 id="title">${title}</h1>
      <div id="content">${showableContent}</div>
      <h2 id="influence_header"></h2>
      <div id="scale-container" style="display: none;">
        <div id="scale" >
          <div class="scale-marker" style="top: 0%;">100%</div>
          <div class="scale-marker" style="top: 20%;">80%</div>
          <div class="scale-marker" style="top: 40%;">60%</div>
          <div class="scale-marker" style="top: 60%;">40%</div>
          <div class="scale-marker" style="top: 80%;">20%</div>
          <div class="scale-marker" style="top: 100%;">0%</div>
        </div>
        <div class="probability-labels-wrapper ">
        </div>
      </div>
  
      <ul id="probabilities" style="list-style-type: none; padding: 0;"></ul>
      <script>
        const vscode = acquireVsCodeApi();
        function resetAllTokens() {
          const probabilityLabels = document.querySelector('.probability-labels-wrapper');
          probabilityLabels.innerHTML = '';
  
          document.querySelectorAll('.token').forEach(token => {
            token.style.backgroundColor = 'transparent';
          });
        }
  
        function handleTokenClick(id) {
          resetAllTokens();
          vscode.postMessage({ command: 'tokenClicked', id: id });
          const scaleContainer = document.getElementById('scale-container');
          scaleContainer.style.display = 'block';
        }
  
        function handleNoInfluence(token) {
          const influenceHeader = document.getElementById('influence_header');
          influenceHeader.textContent = "There are no tokens that influence the token: " + token;
        }
  
        function changeTokenColor(id, color) {
          const tokenElement = document.getElementById('token-' + id);
          if (tokenElement) {
            tokenElement.style.backgroundColor = color;
          } 
          
        }

        function updateInfluenceTitle(selectedToken) {
          const influenceHeader = document.getElementById('influence_header');
          influenceHeader.innerHTML = "Influence on token: <span class='highlighted-token'>" + selectedToken + "</span>";
      
        }


        function updateTokenInfluence(token, probability) {

          if (token !== undefined) {
        
            const labelsWrapper = document.querySelector('.probability-labels-wrapper');
            const newProbabilityDiv = document.createElement('div');
            newProbabilityDiv.className = 'probability-label';

            const newProbabilityDivMarker = document.createElement('div');
            newProbabilityDivMarker.className = 'probability-label-marker';

            const adjustedProbability = (1 - (probability / 100)) * 100;

            newProbabilityDivMarker.style.top = adjustedProbability + "%";

            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = token+ ": " + probability+ "%";
            tooltip.style.display = 'none';

      
            newProbabilityDivMarker.addEventListener('mouseover', () => {
                tooltip.style.display = 'block';
            });

            newProbabilityDivMarker.addEventListener('mouseout', () => {
                tooltip.style.display = 'none';
            });

            newProbabilityDivMarker.appendChild(tooltip);
            if (probability < 0.01) {
              newProbabilityDiv.innerHTML = "Token: <span class='highlighted-token'>" + token + "</span> has less than 0.01% influence";
            } 
            else {
              newProbabilityDiv.innerHTML = "Token: <span class='highlighted-token'>" + token + "</span> has " + probability + "% influence";
            }

            labelsWrapper.append(newProbabilityDiv);
            labelsWrapper.append(newProbabilityDivMarker);
          } 
        }
        window.addEventListener('message', event => {
          const message = event.data;
          switch (message.command) {
            case 'changeColor':
              changeTokenColor(message.id, message.color);
              updateTokenInfluence(message.token, message.probability);
              updateInfluenceTitle(message.selectedToken)
              break; 
            case 'noInfluence': 
              handleNoInfluence(message.token);
              break;

          }
        });
      </script>
    </body>
    </html>
  `;
  }

  //function to display an error message
  public async displayError(err: Error){
    console.log('Error generating code snippet: '+err.message);
    this.showError('Rationale Panel', 'Error generating code snippet: '+err.message);
  }

}

/**
   * Communicates with the backend to obtain output code and related rationales information based on the input code.
   *
   * @param {string} input - The input code.
   */
async function sendCodeToModel(input:string) {
    if (input.length !== 0) {
      let data = {
        "prompt": input,
      };
      try {
        console.log("fetching rationales.");
        let response = await fetch("http://127.0.0.1:5000/prompt", { //change to http://127.0.0.1:5000/prompt if testing locally or /prompt if deploying on Beanstalk
          "method": "POST",
          "headers": { "Content-Type": "application/json" },
          "body": JSON.stringify(data),
        });
        let json = await response.json();
        var rationales = JSON.parse(JSON.stringify(json));
        console.log(rationales);
  
        return rationales;
  
      } catch(e){
        console.log("There was an error with your request: "+(e as Error).stack);
        
        return e;
      }
    }
}
