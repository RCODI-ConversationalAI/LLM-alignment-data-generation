# LLM-alignment-data-generation

## AIM 3.2.1: LLM Alignment with Empathy

### Model Version 1: RLSF

This model improves upon the foundation model by applying 5 symbolic rules (i.e., voice, tense, pronoun, mood, affective words) and augmenting data size.

### Data Sources
We use questions only from the following data sources.

1. **Legal (Tenant Law)**
    - Data: 2,124 unique questions.
    - Dataset link: [Open Australian Legal QA](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa)
2. **Mental Health (Q&A Setting)**
    - Data: 785 unique questions.
    - Dataset link: Questions sourced from [CounselChat](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/public-data/counsel-chat) (Bertagnolli,2020) and [Psych8K](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/public-data/psych8k) (Liuetal.,2023a) .


### Data Generation

We generate responses using GPT-4o in Q&A setting in each data source. Specifically, for each symbolic rule, we generate 2,909 pairs of responses, one the rule applied, the other not, thus a total of 14,545.

### Model Version 2: RLHF
