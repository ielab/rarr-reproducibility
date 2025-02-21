# RARR Unraveled: Component-Level Insights into <br> Hallucination Detection and Mitigation

Welcome to the repository for our reproducibility paper submission to SIGIR ’25.  The original <br> paper by Gao et. al (2023) can be found [here](https://arxiv.org/abs/2210.08726).   

In our work, we critically examine RARR and adapt its framework to incorporate publicly available <br>
evidence retrieval systems and generative models, thereby operationalizing the approach. We focus <br>
on hallucination detection, analyzing how each pipeline component contributes to this task. We conduct <br>
a sentence-level analysis of hallucinations to provide a more granular assessment of RARR’s performance,<br>
identifying insights into RARRs strengths, limitations, and potential areas for improvement. Two key <br>
findings are that query generation and retrieval are effective and that the agreement module is a weak <br>
link in the RARR pipeline.


## **Setup Guide**

### Environment
conda env create -f environment.yaml

### **LLM Promps**
  - [Query Generation](prompts/config_query-gen_Llama_mod.toml)
  - [Iterative Query Generation](prompts/config_query-gen-iterative_Llama_mod.toml)
  - [Agreement](prompts/config_agreement_Llama_mod.toml)
  - [Query-Evidence Relevance Labeling](notebooks/gemini_labelling.ipynb)
  - [Sentence-Query Quality Judgements](notebooks/gemini_labelling.ipynb)
  - [Few Shot Hallucination Detection](/prompts/config_zero-shot-1_Llama.toml)


### Data Preprocessing
#### FAVABench
See fava_datasets.ipynb notebook.

Overview of steps:
1. 'mark' annotations and their tagged content (e.g., `<mark>...</mark>`) are removed.
2. Annotations unrelated to hallucinations (e.g., `<delete>…</delete>`) are removed.
3. Responses are sentence-tokenized using Python’s SpaCy package, and sentences with obvious errors in their annotations (e.g., lone annotation as a sentence, leading end annotation, and dangling start annotation) are fixed.
4. Annotations that remain for each sentence (and do not span multiple sentences) are processed and removed.
5. Sentences were decontextualized as in [Choi et al., 2021](#) using Gemini-1.5-pro-002. This step is performed by prompting an LLM to decontextualize the sentence, given its response.

This results in a dataset of 5,150 decontextualized sentences, each labeled with a count of hallucination types present.

#### WikiBib
See wiki_dataset.ipynb notebook.
#### Wikipedia
**Preprocessing steps:**
1. Chunked into 512-token passages.
2. Overlap of 32 tokens between passages.
## Running Experiments

1. **Configuration Files:**
   - Each experiment uses a configuration file located in the `.configs/` directory.
   - Configuration settings are organized by RARR components:
     - **Query Generation (q)**
     - **Evidence Retrieval (r)**
     - **Agreement (a)**
   - The configuration filenames encode the component settings. For example, a filename like `config_q1_r1_q1.yaml` indicates that:
     - The first query generation experiment is being used.
     - The first evidence retrieval experiment is being used.
     - And so on.

2. **Running an Experiment:**
   - After validating the configuration settings, run the experiment by specifying the configuration file path and the dataset name.
   - For example:

   ```bash
   python run_agreement_modular.py \
     --config_file_path "./configs/config_q6_r1_a1.yaml" \
     --dataset_name "fava"
