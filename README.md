# code-rationales

In recent years, Language Models for Code (LMC) have significantly changed the landscape of software engineering (SE) on downstream tasks, such as code generation, by making software development more efficient. Therefore, a growing interest has emerged in further evaluating these Language Models to homogenize the quality assessment of generated code. As the current evaluation process can significantly overreact on accuracy-based metrics, practitioners often seek methods to _intepret_ LMC outputs beyond canonical benchmarks. While the majority of research reports on code generation effectiveness in terms of expected ground truth, scant attention has been paid to LLMs' explanations. In essence, the decision-making process to generate code is hard to interpret. To bridge this evaluation gap, we introduce _code rationales_ (**CodeQ**), a technique with rigorous mathematical underpinning, to identify subsets of tokens that can explain individual code predictions. We conducted a thorough _Exploratory Analysis_ to demonstrate the method's _applicability_ and a _User Study_ to understand the _usability_ of code-based explanations. Our evaluation demonstrates that \codeRational is a powerful interpretability method to explain how (less) meaningful input concepts (\ie natural language particle `at') highly impact output generation (i.e code conditionals). Moreover, participants of this study highlighted **CodeQ's** ability to show a causal relationship between the input and output of the model with readable and informative explanations on _code completion_ and _test generation_ tasks. Additionally, **CodeQ** also helps to uncover model rationale, facilitating comparison with a human rationale to promote a fair level of trust and distrust in the model.
![formalcodeq2 (1)](https://github.com/user-attachments/assets/580bf7ef-12c8-4ef7-9503-c20df4dc3dba)


![approach-rationales](https://github.com/user-attachments/assets/582968ce-6bae-4877-90b7-6a4691dcc268)

![dependency_map4 (1)](https://github.com/user-attachments/assets/0396e629-f033-4c24-b3f7-8495aa59a5e3)
![ewash2 (1)](https://github.com/user-attachments/assets/d6c68cab-a38f-41ca-b311-51dab472e400)

![experiment_pipeline (1)](https://github.com/user-attachments/assets/a5b3aa79-31dc-4a9a-b7c6-5ed6482e0780)
![fig4_survey_ss](https://github.com/user-attachments/assets/223b19b3-d37f-4897-abe3-fd69e21493e6)
![sec_5_fig_case_study](https://github.com/user-attachments/assets/7ef7c5c0-ecf2-4991-9088-93a39f0a68b3)
