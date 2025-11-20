import * as vscode from "vscode";

/* Stores the rationale data. View the example.json files for practical examples.*/
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

interface colorDict {
    color: string;
    description: string;
    token: string;
}

export class VisualizationPanelProvider implements vscode.WebviewViewProvider {
    
    private _view?: vscode.WebviewView;
    rationaleJsonFile: rationaleDataStructure;
    boxLocations: { [key: string]: number[] };
    private boxColors: {[key: number]: colorDict};
  
    constructor(private readonly _extensionUri: vscode.Uri) {
      this.rationaleJsonFile = {};
      this.boxLocations = {}; //stores the px locations of the concept_view boxes
      this.boxColors = {}; //stores the colors of the token influences
    }


    public resolveWebviewView(webviewView: vscode.WebviewView, context: vscode.WebviewViewResolveContext, token: vscode.CancellationToken){
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        this.updateContent("Visualization", "View the generated text");
    }

    public displayVisualization(highestTokenIds: number[], clickedTokenId: number){

       
        let clickedToken = this.rationaleJsonFile[clickedTokenId]?.token === `\n`? '&#92;n' : this.rationaleJsonFile[clickedTokenId]?.token;

        if (highestTokenIds.length > 0) {      
            this.boxLocations = {};
            this.boxColors = {};
            let naturalLanguage: { [key: string]: string[] } = {};
            let programmingLanguage: { [key: string]: string[] } = {};
            let currToken; 
            for (let i = 0; i < highestTokenIds.length; i++) {
                currToken = this.rationaleJsonFile[highestTokenIds[i]];
                let tmp = "";
                if (currToken?.concept_view[0] === 'Programming Language') {
                    const key = currToken.concept_view[1];
                        if (programmingLanguage[key]) {
                        if (!programmingLanguage[key].includes(currToken.concept_view[2])) {
                            programmingLanguage[key].push(currToken.concept_view[2]);
                            tmp = currToken.concept_view[2];
                        }
                    } else {
                        programmingLanguage[key] = [currToken.concept_view[2]];
                        tmp = currToken.concept_view[2];
                    }
                }
                else if (currToken?.concept_view[0] === 'Natural Language') {
                    const key = currToken.concept_view[1];
                        if (naturalLanguage[key]) {
                        if (!naturalLanguage[key].includes(currToken.concept_view[2])) {
                            naturalLanguage[key].push(currToken.concept_view[2]);
                            tmp = currToken.concept_view[2];
                        }
                    } else {
                        naturalLanguage[key] = [currToken.concept_view[2]];
                        tmp = currToken.concept_view[2];
                    }
                }
                if (typeof currToken !== "undefined") {
                    this.boxColors[i] = {color: this._getRandomColor(), description: currToken.concept_view[2], token: currToken.token};
                }
                
            }

            let conceptView = ``;

            if( Object.keys(naturalLanguage).length === 0){
                conceptView = this.createLanguageDiagram("Programming Language", programmingLanguage, 180, 35)[0];

            }
            else if( Object.keys(programmingLanguage).length === 0){
                conceptView = this.createLanguageDiagram("Natural Language", naturalLanguage, 180, 35)[0];
            }
            else{
                //create both graphs and put them one under another
                let naturalLanguageDiagram, endpoint;
                [naturalLanguageDiagram, endpoint] = this.createLanguageDiagram("Natural Language", naturalLanguage, 180, 35);
                let programmingLanguageDiagram = this.createLanguageDiagram("Programming Language", programmingLanguage, 180, endpoint)[0];
                conceptView = naturalLanguageDiagram + programmingLanguageDiagram;
            }

            conceptView += this.createTokensAndPointers(highestTokenIds);

            if(clickedToken){

                conceptView +=`<rect class="hascolor" width="50" height="15" x="400" y="115" fill="none" stroke="rgba(0,98,114,1)"/> 
                <text x="425" y="125" font-size="8" fill="white" text-anchor="middle">${clickedToken}</text>`;

            }
            
            if(this._view?.webview){
                this.updateContent("Visualization",conceptView);
            } 
        }
        else{
            this.updateContent("Visualization ", `<h3>There are no tokens that influenced the token ${clickedToken} </h3>`);
        }
    }

    private createTokensAndPointers(tokenIds: number []){
        /* Creates the token boxes and arrows that influence the clicked token */
        let tokensAndPointers = `<marker id="arrowhead" class="hascolor" markerWidth="15" markerHeight="10" refX="10" refY="3" orient="auto" viewBox="0 0 10 10">
                                    <path d="M 0 0 L 10 3 L 0 6 Z" stroke="context-stroke" fill="white"/>
                                </marker>`;
        let y = 65;
        for (let i = 0; i < tokenIds.length; i++) {
            let currToken =  this.rationaleJsonFile[tokenIds[i]]?.token === `\n`? '&#92;n' : this.rationaleJsonFile[tokenIds[i]]?.token;
            tokensAndPointers += `<rect class="hascolor" width="50" height="15" x="22" y="${y}" fill="none" stroke="${this.boxColors[i].color}" />
                    <text x="45" y="${y + 12}" font-size="8" fill="white" text-anchor="middle" >
                        ${currToken} 
                    </text>
            `; 

            let conceptViewArr = this.rationaleJsonFile[tokenIds[i]]?.concept_view;
            if(conceptViewArr){
                for(let j = 2; j <= conceptViewArr.length -1; j++) {
                    let boxId = conceptViewArr[j].replaceAll(" ", "-").toLowerCase();
                    let x2 = this.boxLocations[boxId][0]-10 ;
                    let y2 = this.boxLocations[boxId][1] +7;

                    tokensAndPointers +=  ` <line class="hascolor" x1="72" y1="${y+8}" x2="${x2+10}" y2=${y2+1} stroke="${this.boxColors[i].color}" stroke-width="1" marker-end="url(#arrowhead)" />
                    `;
                }
            }
            y += 40;
        }
        
        return tokensAndPointers;
    }

    private _getRandomColor() {
        let g = Math.round, h = Math.random;
        const colorTheme = vscode.window.activeColorTheme.kind;

        switch(colorTheme) { 
            case 1: { 
               console.log("Light theme, Choosing dark color");
               return 'rgba(' + g(h()*64) + ',' + g(h()*64) + ',' + g(h()*64) + ',' + '0.6)';            
            } 
            case 2: { 
               console.log("dark theme, Choosing light color");
               return 'rgba(' + g(h()*64+192) + ',' + g(h()*64+192) + ',' + g(h()*64+192) + ',' + '0.6)';
            } 
            case 3: { 
               console.log("Contrast theme, Choosing dark color");
               return 'rgba(' + g(h()*192+15) + ',' + g(h()*192+15) + ',' + g(h()*192+15) + ',' + '0.6)';
            } 
        } 

        return 'rgba(' + g(h()*192+15) + ',' + g(h()*192+15) + ',' + g(h()*192+15) + ',' + '0.6)';

    }


    private createLanguageDiagram(title: string, dictionary: { [key: string]: string[] }, x: number, y: number): [string, number] {
        const colorTheme = vscode.window.activeColorTheme.kind;


        let color = "";
        if (title === "Natural Language") {
            color = colorTheme === 1 
            ? "rgba(0, 116, 143, 0.8)" : "rgba(0, 207, 255, 0.8)"; 
            console.log("Color for natural language: ", color);
        }
        else {
            color = colorTheme === 1 
            ? "rgba(146, 131, 0, 0.8)" : "rgba(255, 244, 146, 0.8)" ; 
            console.log("Color for programming language: ", color);
        }

        let languageDiagram = '';
        let currY: number = y;
    
        for (const key in dictionary) {
            if (dictionary.hasOwnProperty(key)) {
                let newDiagram = this.createChartWithinConceptView(x, currY + 15, key, dictionary[key], color);
                languageDiagram += newDiagram[0];
                currY = newDiagram[1]; 
            }
            currY += 15;
        }
        let outerHeight = currY - y; 
        
        /* The larger Natural Language/Programming Language boxes */
        languageDiagram += ` 
        <rect class="hascolor" id="lang" width="130" height="${outerHeight}" x="${x - 10}" y="${y}" fill="none" stroke="${color}"/>
        <text x="${x + 55}" y="${y + 10}" font-size="10" font-weight="bold" fill="white" text-anchor="middle">
            ${title}
        </text>`;
    
        return [languageDiagram, currY + 15];
    }
    
    public createChartWithinConceptView(x: number, y: number, title: string, values: string[], color: string): [string, number] {
        /* Creates the inner category boxes for influencing tokens */
        let svgChart = `
        <rect class="hascolor" width="110" height="15" x="${x}" y="${y}" fill="none" stroke="${color}"/>
        <text x="${x + 55}" y="${y + 10}" font-size="8" fill="white" text-anchor="middle" font-style="italic">
            ${title}
        </text>`;
    
        for (let i = 0; i < values.length; i++) {
            let boxId = values[i].replaceAll(" ", "-").toLowerCase();
            y += 15;
            let j = 0;
            let linecolor;
            while (j < Object.keys(this.boxColors).length) {
                if (this.boxColors[j].description === values[i]) {
                    linecolor = this.boxColors[j].color;
                    break;
                }
                else { j++; }
            }
            if (!this.boxLocations[boxId]) {
                this.boxLocations[boxId] = [];
            }
            this.boxLocations[boxId].push(x,y);
            //inner boxes
            svgChart += `<rect class="hascolor" id="${boxId}" width="110" height="15" x="${x}" y="${y+1.5}" fill="none" stroke="var(--vscode-editor-foreground)"/>
            <text x="${x + 55}" y="${y + 10}" font-size="8" fill="var(--vscode-editor-foreground)" text-anchor="middle">
                ${values[i]}
            </text>
            <line class="hascolor" x1="290" y1="${y+8}" x2="400" y2="123" stroke=${linecolor} stroke-width="1" marker-end="url(#arrowhead)" />
            `;
           
        }
        
        return [svgChart, y + 15];
    };

    public clearVisualizationPanel(){
        this.updateContent("Visualization Panel", "Select a token to show visualization");
    }


    public updateRationale(updatedRationaleData: rationaleDataStructure ){
        this.rationaleJsonFile = updatedRationaleData;
    }

    public updateContent(title: string, content: string) {


        if (this._view) {
            if (content === "" ){
                content = "Select text to generate above";
            }
            this._view.webview.html = this._getHtmlForWebview(this._view.webview, title, content);
        }
    }
    private _getHtmlForWebview(webview: vscode.Webview, title: string, content: string): string {
        /* Creats the HTML display in the Visualization panel */
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>${title}</title>
                <style>
                    .button { background-color: white; border: 2px solid green; color: black; }
                    .buttonb { background-color: black; border: 2px solid white; color: white; }
                    .buttonw { background-color: white; border: 2px solid black; color: black; }
                </style>
            </head>
            <body>
                <h1>${title}</h1>
                <button class="button" onClick="changecolor('color')"> Color </button>
                <button class="button buttonb" onClick="changecolor('black')"> Black </button>
                <button class="button buttonw" onClick= "changecolor('white')"> White </button>
                <svg width="500" height="500">
                    ${content}
                </svg>
            <script>
                colorlist = getColors();
                function changecolor(param) {
                    changeboxes(param);
                    changetext(param);
                }

                function changeboxes(param) {
                    const collection = document.querySelectorAll(".hascolor");
                    for (i=0; i<collection.length; i++) {
                        if (param == 'color') {
                            collection[i].setAttribute("stroke", colorlist[i]);
                        }
                        else {
                            collection[i].setAttribute("stroke", param);
                        }
                    }
                }

                function changetext(param) {
                    let color = '';
                    if (param == 'black') {color='gray';} else {color='white';}
                    document.querySelectorAll('text').forEach(text=>{ text.setAttribute("fill",color);});
                }
                function getColors() {
                    let list = [];
                    document.querySelectorAll(".hascolor").forEach(color=> { list.push(color.getAttribute("stroke"));});
                    return list;

                }
            </script>
            </body>
            </html>
        `;
    }
};