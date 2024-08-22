# Why is Accuracy Not Enough for Interpretability? On Rationalizing Language Models For Code 

## Project overview

In recent years, Language Models for Code (LMC) have significantly changed the landscape of software engineering (SE) on downstream tasks, such as code generation, by making software development more efficient. Therefore, a growing interest has emerged in further evaluating these Language Models to homogenize the quality assessment of generated code. As the current evaluation process can significantly overreact on accuracy-based metrics, practitioners often seek methods to _intepret_ LMC outputs beyond canonical benchmarks. While the majority of research reports on code generation effectiveness in terms of expected ground truth, scant attention has been paid to LLMs' explanations. In essence, the decision-making process to generate code is hard to interpret. To bridge this evaluation gap, we introduce _code rationales_ (**CodeQ**), a technique with rigorous mathematical underpinning, to identify subsets of tokens that can explain individual code predictions. We conducted a thorough _Exploratory Analysis_ to demonstrate the method's _applicability_ and a _User Study_ to understand the _usability_ of code-based explanations. Our evaluation demonstrates that **CodeQ** is a powerful interpretability method to explain how (less) meaningful input concepts (i.e. natural language particle `at') highly impact output generation (i.e code conditionals). Moreover, participants of this study highlighted **CodeQ's** ability to show a causal relationship between the input and output of the model with readable and informative explanations on _code completion_ and _test generation_ tasks. Additionally, **CodeQ** also helps to uncover model rationale, facilitating comparison with a human rationale to promote a fair level of trust and distrust in the model.


This repository serves as an online companion to the ICSE '25 paper titled "Why is Accuracy Not Enough for Interpretability? On Rationalizing Language Models For Code." It includes expanded material from the evaluation, as well as links to the data and code. 

Below we provide links to the **CodeQ** artifacts such as experimental notebooks, scripts, survey raw data and analysis, as well as the code repository for our implementation of the Comet both as an extensible Python library. We also explain code rationales approach design and an example of our surver that proves the usability of our approach.

---------



## Code rationales artifacts

| **Artifact**           | **Repository Folder**     | **Description**                                                                                                 |
|------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| _Documented Notebooks_ | experimental_notebooks    | Statistical analysis for _global explanation_ it include extended figures with rationales of different datasets |
| _Source code_          | nbs                       | [Nbdev](https://nbdev.fast.ai/) format notebooks with the code rationales experimentation                       |
| _Source code_          | code_rationales           | Generated code by nbdev as a python library                                                                     |
| _Source code_          | scripts                   | External libraries and utilities for running global experiments                                                 |
| _User study analysis_  | survey-artifacts          | Spreadsheets with participant answers and statistical summarization                                             |
| _Models_               | **_Upon accepted paper_** |                                                                                                                 |
| _Experimental data_    | **_Upon accepted paper_** |                                                                                                                 |

### Documented Notebooks
This folder contains the dataset analysis  with the exploratory analysis for both Natural Language (NL) and Source code (NC)

1. [Exploratory Data Analysis - Concepts Frequency Across Datasets](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/%5B1.1.1%5D_%5B1.1.2%5D.ipynb)
2. [Exploratory Data Analysis - Distribution of Semantic and Non Semantic top rationales](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/%5B1.2.1%5D_%5B1.2.2%5D.ipynb)
3. [Exploratory Data Analysis - Distribution of Rationales](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/%5B1.4.0.1%5D_%5B1.4.0.3%5D.ipynb)
4. [Exploratory Data Analysis - Proportionality of NL and SC](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/%5B1.4.0.1%5D_%5B1.4.0.3%5D.ipynb)
5. [Exploratory Data Analysis - Dependencies between rationales and targets](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/%5B1.4.1%5D_%5B1.4.2%5D_%5B1.4.3%5D.ipynb)

After running the experiments across datasets with the exploratory data analysis we captured different analyses for each datasets. For instance, **capture** folder inside _experimental_notebooks_ contains the result for the comulative rationales probabilities per dataset:

![Distribution](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/captures/distributions/sc/level_1_rationales_distributions.jpg)

As a global analysis for rationales we generated several heatmaps that related the input rationales and generated code combining concepts at the AST level 1 and 2

![heatmap](https://github.com/WM-SEMERU/code-rationales/blob/master/experimental_notebooks/captures/heatmaps/nl_sc/level_2_1.jpg)


### User Study Analysis



---------

## Code rationales approach

**CodeQ** compresses four steps to transform a interpretability tensor from the matrix representation of the input and input set of tokens and their relationship. 



1. **Stating Preconditions**: The first step involves preparing the necessary conditions for using the  method to interpret an LMC. This includes making the model compatible using a specific algorithm and structuring "interpretable concepts," which are meant to introduce understandable ideas related to the model's input-output. These concepts are tailored to the specific software engineering (SE) tasks. Two types of interpretability concepts are proposed: one based on Abstract Syntax Tree (AST) for code generation, and the other on focal methods for test case generation.

2. **Constructing Testbeds**: The second step is about creating testbeds by selecting and configuring the model's input, depending on the SE task and interpretability concepts. For example, prompts are used to create a testbed for code generation, and the generated code is concatenated with the prompt to form a new testbed for applying **CodeQ**, which is referred to as a set of generated snippets.

3. **Building Interpretability Tensors**: The third step involves applying the **CodeQ** method, which is designed to interpret predictions made by language models. **CodeQ** is compatible with both encoder-decoder and decoder-only models and introduces three mathematical components to transform tokens from a snippet into an "interpretability tensor".


![approach-rationales TOSEM drawio](https://github.com/user-attachments/assets/c6ec631b-8f4e-4232-af51-76e906302f06)

4. The interpretability approach uses the tensor $\Phi$ to generate local post-hoc explanations, such as dependency maps. These maps reveal three levels of human-interpretable concepts: 
- $L_1$: fine-grain level rationales, 
- $L_2$: concept rationales, 
- $L_3$: modality.

Additionally, the interpretability tensor can be explored further to generate post-hoc global explanations, with specific statistical analyses.

### Research questions

* **RQ1 [Applicability]:** *How applicable is **CodeQ** to globally interpret code generation?* This RQ explores the use of **CodeQ** in creating understandable explanations for the behavior of language models in code generation tasks. The hypothesis is that greedy rationalization can identify key rationales leading to predictions, providing insights into the model's prediction dependencies.

* **RQ2 [Usability]:** *How useful is **CodeQ** in practical settings?* This RQ assesses the practical usefulness of **CodeQ** through a user study, evaluating qualitative metrics such as usefulness, reliability, readability, and how well **CodeQ** helps in aligning language models.


---------


## Used taxonomy 

We propose two taxonomies $\mathcal{C}$: one for code generation and one for test generation. The first taxonomy is based on Abstract Syntax Trees (ASTs), allowing tokens to be associated with Object-Oriented Programming (OOP) concepts. We also incorporated natural language (NL) concepts using [NLTK](https://www.nltk.org) to map and explain AST nodes like comments and identifiers. The second taxonomy is based on context windows from [eWASH](https://github.com/microsoft/methods2test).

![sec_5_fig_case_study](https://github.com/user-attachments/assets/7ef7c5c0-ecf2-4991-9088-93a39f0a68b3)


---------



## Survey example

![fig4_survey_ss](https://github.com/user-attachments/assets/223b19b3-d37f-4897-abe3-fd69e21493e6)
