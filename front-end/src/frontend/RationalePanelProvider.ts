/* Creates the Rationale panel at the top of the extension.*/ 

import * as vscode from 'vscode';
import { flaskServerRunning , flaskServerReady} from '../extension';
import MaxHeap from 'heap-js';
import { VisualizationPanelProvider } from './VisualizationPanelProvider';
import { json } from 'node:stream/consumers';

const example1 = require('../backend/example1.json');
const example2 = require('../backend/example2.json');
const example3 = require('../backend/example3.json');
const example4 = require('../backend/example4.json');
const example5 = require('../backend/example5.json');
const example6 = require('../backend/example6.json');
const example7 = require('../backend/example7.json');
const example8 = require('../backend/example8.json');
const example9 = require('../backend/example9.json');
const example10 = require('../backend/example10.json');
const example14 = require('../backend/example14.json');

const correct_sample1 = require('../backend/rationale-samples/correct_sample1.json');
const correct_sample2 = require('../backend/rationale-samples/correct_sample2.json');
const correct_sample3 = require('../backend/rationale-samples/correct_sample3.json');
const correct_sample4 = require('../backend/rationale-samples/correct_sample4.json');
const correct_sample5 = require('../backend/rationale-samples/correct_sample5.json');
const correct_sample6 = require('../backend/rationale-samples/correct_sample6.json');
const correct_sample7 = require('../backend/rationale-samples/correct_sample7.json');
const correct_sample8 = require('../backend/rationale-samples/correct_sample8.json');
const correct_sample9 = require('../backend/rationale-samples/correct_sample9.json');
const correct_sample10 = require('../backend/rationale-samples/correct_sample10.json');
const correct_sample11 = require('../backend/rationale-samples/correct_sample11.json');
const correct_sample12 = require('../backend/rationale-samples/correct_sample12.json');
const correct_sample13 = require('../backend/rationale-samples/correct_sample13.json');
const correct_sample14 = require('../backend/rationale-samples/correct_sample14.json');

const incorrect_sample1 = require('../backend/rationale-samples/incorrect_sample1.json');
const incorrect_sample2 = require('../backend/rationale-samples/incorrect_sample2.json');
const incorrect_sample3 = require('../backend/rationale-samples/incorrect_sample3.json');
const incorrect_sample4 = require('../backend/rationale-samples/incorrect_sample4.json');
const incorrect_sample5 = require('../backend/rationale-samples/incorrect_sample5.json');
const incorrect_sample6 = require('../backend/rationale-samples/incorrect_sample6.json');
const incorrect_sample7 = require('../backend/rationale-samples/incorrect_sample7.json');
const incorrect_sample8 = require('../backend/rationale-samples/incorrect_sample8.json');
const incorrect_sample9 = require('../backend/rationale-samples/incorrect_sample9.json');
const incorrect_sample10 = require('../backend/rationale-samples/incorrect_sample10.json');


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
          if(newRationale.length > this._max_input_length){
            this.displayError(new Error("The input code exceeds the maximum length."));
            return;
          }else{
            this.showError('Rationale Panel',this.getUserHighlightedCode());
          }
        }

        console.log("Current rationale", newRationale);
        
        if([
          "example1","example2","example3", "example4", "example5", "example6", "example7", "example8", "example9", "example10","example14",
          "correct_sample1","correct_sample2","correct_sample3","correct_sample4","correct_sample5","correct_sample6","correct_sample7",
          "correct_sample8","correct_sample9","correct_sample10","correct_sample11","correct_sample12","correct_sample13","correct_sample14",
          "incorrect_sample1","incorrect_sample2","incorrect_sample3","incorrect_sample4","incorrect_sample5",
          "incorrect_sample6","incorrect_sample7","incorrect_sample8","incorrect_sample9","incorrect_sample10"
        ].includes(newRationale)){
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
            case "example4": {
              this.rationaleJsonFile = example4;
              break;
            }
            case "example5": {
              this.rationaleJsonFile = example5;
              break;
            }
            case "example6": {
              this.rationaleJsonFile = example6;
              break;
            }
            case "example7": {
              this.rationaleJsonFile = example7;
              break;
            }
            case "example8": {
              this.rationaleJsonFile = example8;
              break;
            }
            case "example9": {
              this.rationaleJsonFile = example9;
              break;
            }
            case "example10": {
              this.rationaleJsonFile = example10;
              break;
            }
            case "example14": {
              this.rationaleJsonFile = example14;
              break;
            }
            case "correct_sample1": {
              this.rationaleJsonFile = correct_sample1;
              break;
            }
            case "correct_sample2": {
              this.rationaleJsonFile = correct_sample2;
              break;
            }
            case "correct_sample3": {
              this.rationaleJsonFile = correct_sample3;
              break;
            }
            case "correct_sample4": {
              this.rationaleJsonFile = correct_sample4;
              break;
            }
            case "correct_sample5": {
              this.rationaleJsonFile = correct_sample5;
              break;
            }
            case "correct_sample6": {
              this.rationaleJsonFile = correct_sample6;
              break;
            }
            case "correct_sample7": {
              this.rationaleJsonFile = correct_sample7;
              break;
            }
            case "correct_sample8": {
              this.rationaleJsonFile = correct_sample8;
              break;
            }
            case "correct_sample9": {
              this.rationaleJsonFile = correct_sample9;
              break;
            }
            case "correct_sample10": {
              this.rationaleJsonFile = correct_sample10;
              break;
            }
            case "correct_sample11": {
              this.rationaleJsonFile = correct_sample11;
              break;
            }
            case "correct_sample12": {
              this.rationaleJsonFile = correct_sample12;
              break;
            }
            case "correct_sample13": {
              this.rationaleJsonFile = correct_sample13;
              break;
            }
            case "correct_sample14": {
              this.rationaleJsonFile = correct_sample14;
              break;
            }
            case "incorrect_sample1": {
              this.rationaleJsonFile = incorrect_sample1;
              break;
            }
            case "incorrect_sample2": {
              this.rationaleJsonFile = incorrect_sample2;
              break;
            }
            case "incorrect_sample3": {
              this.rationaleJsonFile = incorrect_sample3;
              break;
            }
            case "incorrect_sample4": {
              this.rationaleJsonFile = incorrect_sample4;
              break;
            }
            case "incorrect_sample5": {
              this.rationaleJsonFile = incorrect_sample5;
              break;
            }
            case "incorrect_sample6": {
              this.rationaleJsonFile = incorrect_sample6;
              break;
            }
            case "incorrect_sample7": {
              this.rationaleJsonFile = incorrect_sample7;
              break;
            }
            case "incorrect_sample8": {
              this.rationaleJsonFile = incorrect_sample8;
              break;
            }
            case "incorrect_sample9": {
              this.rationaleJsonFile = incorrect_sample9;
              break;
            }
            case "incorrect_sample10": {
              this.rationaleJsonFile = incorrect_sample10;
              break;
            }
          }

        }
        else{
          await flaskServerReady;
          console.log("\x1b[0mFlask server is ready, sending user selected code (",newRationale,") to model\x1b[0m");
          
          let results = await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Generating code snippet.",
            cancellable: false
          }, (progress, token) => {
            return sendCodeToModel(newRationale); 
          });
          
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
    
    for(let i = 0; i < 50; i++){
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

          let p = topProbability.probability * 100;
          let hue = (1 - Math.max(0, Math.min(1, p / 100))) * 240;
          let influenceColor = `hsl(${hue}, 100%, 50%)`;

          this._view?.webview.postMessage({
            command: 'changeColor',
            id: String(topProbability.index),
            color: influenceColor, 
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

        function pickTextColor(bgColor) {
          function hslToRgb(h, s, l) {
            s /= 100; l /= 100;
            const c = (1 - Math.abs(2*l - 1)) * s;
            const x = c * (1 - Math.abs(((h/60) % 2) - 1));
            const m = l - c/2;
            let r=0,g=0,b=0;
            if (0 <= h && h < 60) { r=c; g=x; b=0; }
            else if (60 <= h && h < 120) { r=x; g=c; b=0; }
            else if (120 <= h && h < 180) { r=0; g=c; b=x; }
            else if (180 <= h && h < 240) { r=0; g=x; b=c; }
            else if (240 <= h && h < 300) { r=x; g=0; b=c; }
            else { r=c; g=0; b=x; }
            r = Math.round((r+m)*255);
            g = Math.round((g+m)*255);
            b = Math.round((b+m)*255);
            return [r,g,b];
          }

          const m = bgColor.match(/hsl\\(\\s*([\\d.]+)\\s*,\\s*([\\d.]+)%\\s*,\\s*([\\d.]+)%\\s*\\)/i);
          if (!m) return "white";

          const h = parseFloat(m[1]);
          const s = parseFloat(m[2]);
          const l = parseFloat(m[3]);
          const rgb = hslToRgb(h,s,l);
          const r = rgb[0];
          const g = rgb[1];
          const b = rgb[2];

          const luminance = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
          return luminance > 0.6 ? "black" : "white";
        }

        function updateTokenInfluence(token, probability, color) {
          if (token !== undefined) {
            const labelsWrapper = document.querySelector('.probability-labels-wrapper');
            const newProbabilityDiv = document.createElement('div');
            newProbabilityDiv.className = 'probability-label';

            const newProbabilityDivMarker = document.createElement('div');
            newProbabilityDivMarker.className = 'probability-label-marker';

            const adjustedProbability = (1 - (probability / 100)) * 100;
            newProbabilityDivMarker.style.top = adjustedProbability + "%";
            newProbabilityDivMarker.style.borderTopColor = color;

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

            const textColor = pickTextColor(color);
            const style = "background-color: " + color + "; border-color: " + color + "; color: " + textColor + ";";

            if (probability < 0.01) {
              newProbabilityDiv.innerHTML = "Token: <span class='highlighted-token' style='" + style + "'>" + token + "</span> has less than 0.01% influence";
            } 
            else {
              newProbabilityDiv.innerHTML = "Token: <span class='highlighted-token' style='" + style + "'>" + token + "</span> has " + probability + "% influence";
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
              updateTokenInfluence(message.token, message.probability, message.color);
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

  public async displayError(err: Error){
    console.log('Error generating code snippet: '+err.message);
    this.showError('Rationale Panel', 'Error generating code snippet: '+err.message);
  }

}

async function sendCodeToModel(input:string) {
    if (input.length !== 0) {
      let data = {
        "prompt": input,
      };
      try {
        console.log("fetching rationales.");
        let response = await fetch("http://127.0.0.1:5000/prompt", {
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