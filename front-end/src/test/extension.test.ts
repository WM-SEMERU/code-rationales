const assert = require('assert');
const vscode = require('vscode');
import { deactivate, startFlaskServer } from '../extension';


suite('Extension Test Suite', () => {
	vscode.window.showInformationMessage('Start all tests.');

	test('BlackBoxTest', async () => {
		const executionResult = await vscode.commands.executeCommand('rationale-panel'); // Run command
        assert.ok(executionResult !== undefined, 'Could not run rationale-panel');
	});

  	test("DataDrivenTest", async () => {
    	const activeEditor = vscode.window.activeTextEditor; // Create example text file
        assert.ok(activeEditor, 'No active editor found.');

        await activeEditor.edit((editString: string) => { // Set some highlighted text (Hopefully this works, I have tried a million different options)
            editString.replace(activeEditor.selection, 'example1');
        });

        const highlightedText = activeEditor.document.getText(activeEditor.selection);
        assert.strictEqual(highlightedText, 'example1', '"example1" was not highlighted');

        const fetchedJson = require('../backend/example1.json'); // Grab JSON
        assert.ok(fetchedJson, 'Unable to grab JSON');
        assert.strictEqual(fetchedJson._phrase, 'The Supreme Court on Tuesday rejected a challenge to the constitutionality of the death penalty', 'Returned incorrect JSON.');
  	});

  	test("FunctionalTesting/GlassBooxTest/LogicDrivenTest", async () => {
		startFlaskServer(); // Check if the flask server is being started properly
		setTimeout(() => { // If doesn't start after 500, fail
		  assert.strictEqual(deactivate, 0);
		}, 500);
	  });
   	});
