# Why is Accuracy Not Enough for Interpretability? On Rationalizing Language Models For Code 

In recent years, Language Models for Code (LMC) have significantly changed the landscape of software engineering (SE) on downstream tasks, such as code generation, by making software development more efficient. Therefore, a growing interest has emerged in further evaluating these Language Models to homogenize the quality assessment of generated code. As the current evaluation process can significantly overreact on accuracy-based metrics, practitioners often seek methods to _intepret_ LMC outputs beyond canonical benchmarks. While the majority of research reports on code generation effectiveness in terms of expected ground truth, scant attention has been paid to LLMs' explanations. In essence, the decision-making process to generate code is hard to interpret. To bridge this evaluation gap, we introduce _code rationales_ (**CodeQ**), a technique with rigorous mathematical underpinning, to identify subsets of tokens that can explain individual code predictions. We conducted a thorough _Exploratory Analysis_ to demonstrate the method's _applicability_ and a _User Study_ to understand the _usability_ of code-based explanations. Our evaluation demonstrates that **CodeQ** is a powerful interpretability method to explain how (less) meaningful input concepts (i.e. natural language particle `at') highly impact output generation (i.e code conditionals). Moreover, participants of this study highlighted **CodeQ's** ability to show a causal relationship between the input and output of the model with readable and informative explanations on _code completion_ and _test generation_ tasks. Additionally, **CodeQ** also helps to uncover model rationale, facilitating comparison with a human rationale to promote a fair level of trust and distrust in the model.

This repository serves as an online companion to the ICSE '25 paper titled "Why is Accuracy Not Enough for Interpretability? On Rationalizing Language Models For Code." It includes expanded material from the evaluation, as well as links to the data and code. 

Below we provide links to the **CodeQ** artifacts such as experimental notebooks, scripts, survey raw data and analysis, as well as the code repository for our implementation of the Comet both as an extensible Python library. We also explain code rationales approach design and an example of our surver that proves the usability of our approach.

---------



## Code rationales artifacts

| **Artifact**           | **Repository Folder**     | **Description**                                                                                                 |
|------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| _Documented Notebooks_ | result_analysis    | Statistical analysis for _global explanation_ it include extended figures with rationales of different datasets |
| _Source code_          | nbs                       | [Nbdev](https://nbdev.fast.ai/) format notebooks with the code rationales experimentation                       |
| _Source code_          | code_rationales           | Generated code by nbdev as a python library                                                                     |
| _Source code_          | scripts                   | External libraries and utilities for running global experiments                                                 |
| _User study analysis_  | survey-artifacts          | Spreadsheets with participant answers and statistical summarization                                             |
| _Models_               | **_Upon accepted paper_** |                                                                                                                 |
| _Experimental data_    | **_Upon accepted paper_** |                                                                                                                 |

### Documented Notebooks
This folder contains the dataset analysis  with the exploratory analysis for both Natural Language (NL) and Source code (NC) for code and test generation.

1. [Exploratory Data Analysis - Frequency of NL and SC](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/1_frequency_nl_sc.ipynb)
2. [Exploratory Data Analysis - Distribution of Meaningful and Meaningless Concepts](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/2_distribution_meaningful_meaningless_rationales.ipynb)
3. [Exploratory Data Analysis - Distribution of Rationales Probabilities Across Different Datasets](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/3_distribution_rationales.ipynb)
4. [Exploratory Data Analysis - Proportionality of NL and SC](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/4_proportionality_nl_sc.ipynb)
5. [Exploratory Data Analysis - Dependencies between rationales and targets](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/5_dependencies_between_rationales_targets.ipynb)
6. [Test generation - Local Analysis](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/test_generation/4_local_rationales.ipynb)
7. [Test generation - Global Analyisis](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/test_generation/3_global_statistics_ratio_ewash.ipynb)

After running the experiments across datasets with the exploratory data analysis we captured different analyses for each datasets. For instance, **capture** folder on earch SE tasks for _rq1_exploratory_analysis_ we find the global analysis for code completion and test generation:



As a global analysis for rationales we generated several heatmaps that related the input rationales and generated code combining concepts at the AST level 1 and 2

![heatmap](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/captures/heatmaps/nl_sc/level_2_1.jpg)


Global Analysis for test generation using eWASH. 

![test-generation-heatmap](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/test_generation/captures/6_source_target_heatmap.png)

### User Study Analysis

This folder contains raw data from our user study and CSVs where we aggregated the results and performed statistical analysis based on our research questions.
1. [Raw data of user responses in CSV format](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq2_user_study/CodeRationalSurveyResponses.csv)
2. [Collection of all the user responses and statistical analysis from Qualtrics](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq2_user_study/ResponseForEachQuestion.pdf)
3. [Taxonomy of error cases analysis](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq2_user_study/Errors%20Taxonomy%20-%20Samplings.xlsx)
4. [Survey Evaluation based on our metrics including demographic information](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq2_user_study/Survey_evaluation.xlsx)


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


## Intepretability Concepts $\mathcal{C}$ for code generation

We propose two taxonomies $\mathcal{C}$: one for code generation and one for test generation. The first taxonomy is based on Abstract Syntax Trees (ASTs), allowing tokens to be associated with Object-Oriented Programming (OOP) concepts. We also incorporated natural language (NL) concepts using [NLTK](https://www.nltk.org) to map and explain AST nodes like comments and identifiers. The second taxonomy is based on context windows from [eWASH](https://github.com/microsoft/methods2test). 

The following figure ilustrates on the left (1) case(a) code generation example and the erroneous generated code, case (b) a test generation example. On the right (2) interpretability concepts for code generation.

![sec_5_fig_case_study](https://github.com/user-attachments/assets/7ef7c5c0-ecf2-4991-9088-93a39f0a68b3)


---------
## Dataset analysis


![Distribution](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/code_completion/captures/distributions/sc/level_0_rationales_distributions.jpg)

![test-generation-method-size](https://github.com/WM-SEMERU/code-rationales/blob/master/results_analysis/rq1_exploratory_analysis/test_generation/captures/1_focal_method_size.png)
---------


## Survey example

![fig4_survey_ss](https://github.com/user-attachments/assets/223b19b3-d37f-4897-abe3-fd69e21493e6)
