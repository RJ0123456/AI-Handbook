# Chapter 4: Natural Language Processing

## What Is Natural Language Processing?

Natural Language Processing (NLP) is the branch of AI concerned with enabling computers to understand, interpret, generate, and interact using human language — text and speech.

Language is deeply complex: it is ambiguous, context-dependent, evolving, and culturally rich. NLP bridges the gap between unstructured human language and the structured data that computers process.

## Core NLP Tasks

| Task | Description | Example |
|------|-------------|---------|
| Text Classification | Assign a label to text | Spam detection, sentiment analysis |
| Named Entity Recognition (NER) | Identify entities in text | People, places, organizations |
| Part-of-Speech Tagging | Label words with grammatical roles | Noun, verb, adjective |
| Dependency Parsing | Analyze grammatical structure | Subject-verb-object relationships |
| Machine Translation | Translate between languages | English → French |
| Summarization | Condense long text | News article → 3-sentence summary |
| Question Answering | Answer questions from a passage | Open-domain QA, reading comprehension |
| Text Generation | Produce coherent text | Chatbots, code generation, creative writing |
| Coreference Resolution | Identify when expressions refer to the same entity | "Alice said she was tired." (she = Alice) |

## The NLP Pipeline

```
Raw Text
   ↓
Tokenization      (split into words/subwords/characters)
   ↓
Normalization     (lowercasing, removing punctuation)
   ↓
Stop Word Removal (optional — remove common words)
   ↓
Stemming/Lemmatization (reduce words to base form)
   ↓
Feature Extraction / Embedding
   ↓
Model
   ↓
Output
```

## Text Representations

### Bag of Words (BoW)

Represent text as a vector of word counts. Simple but loses word order and context.

### TF-IDF (Term Frequency–Inverse Document Frequency)

Weights words by how often they appear in a document relative to the corpus. Better than raw counts for distinguishing important terms.

### Word Embeddings

Dense vector representations that capture **semantic meaning**. Similar words have similar vectors.

| Model | Year | Key Idea |
|-------|------|----------|
| Word2Vec | 2013 | Predict surrounding words (CBOW/Skip-gram) |
| GloVe | 2014 | Global word co-occurrence statistics |
| FastText | 2016 | Subword embeddings; handles morphology |

### Contextual Embeddings (Transformers)

Unlike static embeddings, contextual models produce different vectors for the same word in different contexts.

- **BERT:** Bidirectional; great for understanding tasks
- **GPT series:** Left-to-right; great for generation tasks
- **T5, BART:** Encoder-decoder; great for translation and summarization

## Language Models

A **language model** assigns a probability to a sequence of words:

```
P("The cat sat on the mat")
```

Modern **large language models (LLMs)** are trained on vast amounts of text and can perform a wide range of tasks with few or no examples (zero-shot / few-shot learning).

### Prompt Engineering

The practice of crafting inputs to elicit the desired output from an LLM.

**Key techniques:**
- **Zero-shot prompting:** Describe the task and ask for the output directly.
- **Few-shot prompting:** Provide a few examples before the actual query.
- **Chain-of-thought (CoT):** Ask the model to "think step by step."
- **Role prompting:** Set a persona ("You are an expert software engineer...").
- **Structured output:** Request JSON, markdown tables, or other formats.

## Retrieval-Augmented Generation (RAG)

RAG combines a retrieval system with a generative model to ground responses in specific documents:

```
Query → Retriever (find relevant documents) → LLM (generate answer using retrieved context)
```

**Benefits:**
- Reduces hallucinations
- Allows knowledge updates without retraining
- Enables citation of sources

## Key Evaluation Metrics

| Metric | Task | Description |
|--------|------|-------------|
| Accuracy / F1 | Classification, NER | Standard classification metrics |
| BLEU | Machine Translation | N-gram overlap with reference translations |
| ROUGE | Summarization | Recall-oriented n-gram overlap |
| Perplexity | Language Modeling | How well the model predicts held-out text |
| BERTScore | Generation | Semantic similarity using BERT embeddings |
| Human Evaluation | All generative tasks | Gold standard; expensive |

## Common Libraries and Tools

| Library | Purpose |
|---------|---------|
| NLTK | Classical NLP: tokenization, POS tagging, parsing |
| spaCy | Industrial-strength NLP pipeline |
| Hugging Face Transformers | Pre-trained models, fine-tuning |
| Hugging Face Datasets | Standardized datasets |
| LangChain / LlamaIndex | LLM orchestration, RAG pipelines |
| OpenAI API | Access to GPT models |
| Sentence-Transformers | Sentence embeddings, semantic search |

## Summary

NLP has been transformed by the Transformer architecture and large language models. From classical pipelines using tokenization and TF-IDF to modern LLMs capable of nuanced reasoning, the field spans a wide spectrum of techniques. Prompt engineering and RAG have become essential skills for practitioners working with LLMs.

---

**Previous:** [Chapter 3 – Deep Learning](03-deep-learning.md)  
**Next:** [Chapter 5 – Computer Vision](05-computer-vision.md)
