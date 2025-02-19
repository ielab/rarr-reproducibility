# Reproduction of RARR

## Data Preprocessing
### FAVABench

**Preprocessing steps:**
1. 'mark' annotations and their tagged content (e.g., `<mark>...</mark>`) are removed.
2. Annotations unrelated to hallucinations (e.g., `<delete>…</delete>`) are removed.
3. Responses are sentence-tokenized using Python’s SpaCy package, and sentences with obvious errors in their annotations (e.g., lone annotation as a sentence, leading end annotation, and dangling start annotation) are fixed.
4. Annotations that remain for each sentence (and do not span multiple sentences) are processed and removed.
5. Sentences were decontextualized as in [Choi et al., 2021](#) using Gemini-1.5-pro-002. This step is performed by prompting an LLM to decontextualize the sentence, given its response.

This results in a dataset of 5,150 decontextualized sentences, each labeled with a count of hallucination types present.

### WikiBib
**Preprocessing steps:**
1. ???

### Wikipedia
**Preprocessing steps:**
1. Chunked into 512-token passages.
2. Overlap of 32 tokens between passages.
