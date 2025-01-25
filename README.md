# LLM-alignment-data-generation

## AIM 3.2.1: LLM Alignment with Empathy

### Model Version 1: RLSF

This model improves upon the foundation model by applying 5 symbolic rules (i.e., voice, tense, pronoun, mood, emotional/cognitive words) and augmenting data size.

### Data Sources
We use questions only from the following data sources.

1. **Legal (Tenant Law)**
    - Data: ChatGPT-4 generated dataset with 2,000 questions, each paired with accepted or rejected answers based on the four rules.
    - Dataset link: [Open Australian Legal QA](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa)
2. **Mental Health (Q&A Setting)**
    - Data: Questions sourced from [CounselChat](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/public-data/counsel-chat) (Bertagnolli,2020) and [Psych8K](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/public-data/psych8k) (Liuetal.,2023a) .
    - Answers generated for each rule using GPT-4o and Claude 3.5 Sonnet.
    - Lastly, Miracle LLM

### Data Generation

We generate responses using GPT-4, GPT-4o, GPT-3.5 (InstructGPT) and Claude 3.5 Sonnet in Q&A setting in each data source. Specifically, for each symbolic rule, we generate 2,000 pairs of responses, one the rule applied, the other not. 10,000 samples generated for each data source, thus a total of 20,000.

We also applied 


### Model Version 2: RLHF
