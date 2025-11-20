import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { RationalePanelProvider  } from './frontend/RationalePanelProvider';
import { spawn, ChildProcess } from "child_process";
import { VisualizationPanelProvider } from './frontend/VisualizationPanelProvider';

const flaskAppPath = path.join(
  __dirname,
  "..",
  "src",
  "backend",
  "application.py"
);
let flaskServer: ChildProcess | undefined;
let flaskServerRunning: boolean = false;
let serverReadyResolve: (() => void) | undefined;

//wait until flask server is ready
export const flaskServerReady = new Promise<void>((resolve) => {
  serverReadyResolve = resolve;
});

//start Flask server
export function startFlaskServer() {
  if (!flaskServer) {
    flaskServer = spawn("python", [flaskAppPath]);
    flaskServer.stdout?.on("data", (data) => {
      console.log(`Python output: ${data}`);
      if (data.toString().includes("Serving Flask app")) {
        flaskServerRunning = true;
        console.log("Flask server is now running and ready.");
        if (serverReadyResolve) {
          serverReadyResolve();
        }
      }
    });

    flaskServer.stderr?.on("data", (data) => {
      console.error(`Python error: ${data}`);
    });

    flaskServer.on("close", (code) => {
      console.log(`Python process exited with code ${code}`);
      flaskServerRunning = false;
    });
  }
}


export function activate(context: vscode.ExtensionContext) {

  startFlaskServer();

  //display loading messsage for flask server
  vscode.window.withProgress({
    location: vscode.ProgressLocation.Notification,
    title: "Starting flask server.",
    cancellable: false
  }, (progress, token) => {

    return flaskServerReady;

  }).then(()=>{
    if(flaskServerRunning){
      vscode.window.showInformationMessage("Flask server is ready.");
    }else{
      vscode.window.showErrorMessage("The flask server did not start.");
    }
  });

  const visProvider = new VisualizationPanelProvider(context.extensionUri);
  const rationaleProvider = new RationalePanelProvider(context.extensionUri,visProvider);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('rationale-panel', rationaleProvider),
    vscode.window.registerWebviewViewProvider('visualization-panel', visProvider)
  );


  context.subscriptions.push(
    // Command that handles opening and closing the panel
    vscode.commands.registerCommand('rationale-panel', () => {
      vscode.commands.executeCommand('rationale-panel.focus');
    })
    ,vscode.commands.registerCommand('visualization-panel', () => {
      vscode.commands.executeCommand('visualization-panel.focus');
    })
  );

}


export function deactivate() {
  if (flaskServer) {
    flaskServer.kill();
    console.log("Python process killed as extension is deactivated.");
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
    console.log(input);
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
        return;
      }
    } else {
      console.log("There was no input code.  Using example data.");
    }
  }

export { flaskServerRunning };
