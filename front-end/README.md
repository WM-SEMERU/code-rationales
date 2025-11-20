# rationales README
# Project 1: Rationales Visualization (Fall 2024)

An interactive data visualization tool that showcases code generation rationales from Large Language Models. The tool highlights the influences of a selected token in the Rationale Panel and a diagram of how those influences shape the selected token in the Visualization Panel.

This tool uses the sequential-rationales repository by Keyon Vafa, Yuntian Deng, David M. Blei, and Alexander M. Rush, which can be found here: https://github.com/keyonvafa/sequential-rationales


## Installing Dependencies

Before you run the extension, download Node.js at `https://nodejs.org/en`.

After Node.js is installed, run the following commands in the repository folder to install the other dependencies:

`npm install`
`npm install heap-js`

Note the requirements.txt file. You may either install all packages manually or use 

`pip install -r requirements.txt`

**Note: You must have Python 3.11 or earlier installed to run Flask and Torch. Download Python 3.11 here: `https://www.python.org/downloads/release/python-3110/`.

Once all the dependencies are downloaded, then click Run in the Debug environment on VSCode's toolbar. A new window will pop up, where you should highlight some text.
Then click the rationale and visualization button in the activity bar. If you want to run example data, highlight the word example1, example2, or example3. Another window should open where there will be generated text based on the text you highlighted. Click on a token to see the rationales and a visualization. 

## Running the Extension

After you have installed the proper dependencies and cloned the repository (or downloaded the extension from the VSCode marketplace), please activate the extension by opening a new window in VSCode or restarting VSCode. Then, as you generate code, select the generated code and click on the Rationales Extension window. This will pull up the rationales shortly, at which point you can interact with them.


## Video of Usage

https://github.com/user-attachments/assets/31ed267d-ee48-4c81-a668-48c0aa02836b


## Extension Structure

The frontent is built using VSCode's Webview API. It consists of two main components: RationalePanelProvider and VisualizationPanelProvider. Each resolves the webview for their specific panel and visualizes the data sent from the backend in either a selectable rationale breakdown or a graph of rationale connections.

The backend is powered by a Python Flask Server. It handles the server logic and generates the rationale data when requested.


**Enjoy!**
