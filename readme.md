# RAG System with Google Colab, FAISS, and Gemini API

A Retrieval-Augmented Generation (RAG) system that processes structured documentation, creates vector embeddings, and provides an interactive query interface with Google's Gemini AI.

## Features

- **Structure-Aware Document Processing**: Supports DITA maps, Markdown, HTML, and preserves document hierarchy.
- **FAISS Vector Search**: Fast similarity search using Facebook's FAISS library.
- **Gemini AI**: Google's Gemini 2.5 Flash model for intelligent responses.
- **Interactive UI**: Jupyter widgets for easy query testing.
- **Persistent Storage**: Saves indexes to Google Drive for reuse across sessions.
- **Configurable Retrieval**: Adjustable similarity thresholds, top-K results, etc.

## Prerequisites

You need a Google account with access to [Google Colab](https://colab.research.google.com/), a [Gemini API key](https://aistudio.google.com/app/apikey), and source documents (DITA, Markdown, HTML) stored in Google Drive.

**Note**: The [GitHub repo](https://github.com/shealy2020/colab-gdrive-rag-ui-1) has sample documents that you can upload to your Google Drive.

## Getting Started

Follow these steps to set up and run your notebook.

### 1. Create Your Google Colab Notebook

Go to [https://colab.research.google.com/](https://colab.research.google.com/), then create a notebook via the **File** menu.

### 2. Set Up API Key

The code requires a **Gemini API Key** to communicate with the model. Store this key securely in Colab's **Secrets** tool.

1.  Get your key from Google AI Studio: [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key)
2.  In the Colab Notebook, look for the **Secrets** tab in the left sidebar.
3.  Click the **`+`** icon to add a new secret.
4.  Set the **Name** exactly to: `GEMINI_API_KEY`
5.  Set the **Value** to the key you copied in Step 1.
6.  Ensure the "Notebook access" toggle is **ON** for this secret.

### 3. Prepare Source Folders and Documents

1. Create this directory structure in your Google Drive:

   *  `My Drive/gemini-source-index/rag_docs_structured`
        
        This is where you will upload DITA, Markdown, and HTML files.


    * `My Drive/gemini-source-index/rag_index_gemini_faiss`
        
        This is where the FAISS vector index will be saved.

    **Important**:  Update the paths in **Cell 2** if you use different folder names:

   ```python
   DOCS_DIR = '/content/drive/[your custom directory path]'
   FAISS_INDEX_PATH = '/content/drive/MyDrive/[your custom directory path]'
   ```

### 4. Upload source documents (.dita, .md, .html) to `My Drive/gemini-source-index/rag_docs_structured`.      

### 5. Run Cells

Run each [cell](https://github.com/shealy2020/colab-gdrive-rag-ui-1/blob/main/colab-gdrive-rag-ui-1.py) in your notebook sequentially. Alternately, run the entire Python script as a single cell. (I prefer to run each functional block of code separately for troubleshooting purposes.)

**Important**: *Prior to running the script, uncomment lines 5 and 6 of Cell 1A if your notebook does not already have these Python libraries loaded.* 

### Step 5: Run notebook.

Execute cells in order:
1. [Cell 1A](../../../posts/colab-gdrive-rag-ui/#cell-1a): Installs dependencies.
2. [Cell 1B](../../../posts/colab-gdrive-rag-ui/#cell-1b): Imports libraries and validate API key.
3. [Cell 2](../../../posts/colab-gdrive-rag-ui/#cell-2): Mounts Google Drive and configures paths.
4. [Cell 3](../../../posts/colab-gdrive-rag-ui/#cell-3): LlamaIndex structure-aware document loading and chunking.
5. [Cell 4](../../../posts/colab-gdrive-rag-ui/#cell-4): Generates embeddings and builds FAISS index.
6. [Cell 5](../../../posts/colab-gdrive-rag-ui/#cell-5): Launches interactive query interface.

### Step 6: Enter and submit your query.

**Note**: The LLM will only return response based on the source you provide. 





## Cell Descriptions

### Cell 1A
**Purpose**: Installs all required Python packages before importing them.

**What it does**:
- Installs FAISS (Facebook AI Similarity Search) for vector indexing.
- Installs Sentence Transformers for text embeddings.
- Installs Google Generative AI SDK (Gemini API client).
- Installs LlamaIndex for document processing.
- Installs supporting libraries (numpy, lxml, ipywidgets).

**Important**: *Prior to running the script, uncomment lines 5 and 6 of Cell 1A if your notebook does not already have these Python libraries loaded.*  

---

### Cell 1B
**Purpose**: Imports all installed packages and validates your Gemini API key.

**What it does**:
- Imports core libraries (FAISS, transformers, Gemini client).
- Retrieves your `GEMINI_API_KEY` from Colab Secrets.
- Validates the API key is accessible.
- Sets up the environment for subsequent cells.

---

### Cell 2
**Purpose**: Connects to Google Drive and creates source and indexing directories or verifies that these directories exist.

**What it does**:
- Mounts your Google Drive to the Colab runtime.
- Defines paths:
  - `DOCS_DIR`: Where your source documents are stored.
  - `FAISS_INDEX_PATH`: Where the vector index will be saved.
- Creates directories if they don't exist.
- Validates the file structure.

---

### Cell 3
**Purpose**: Loads your documents, parses DITA map structure, and splits content into chunks.

**What it does**:
1. **DITA Map Parsing**: 
   - Recursively parses `.ditamap` files to extract document hierarchy.
   - Builds a path map (e.g., "Manual > Chapter 1 > Topic Name").
   - Preserves structural context for better retrieval.

2. **Document Loading**:
   - Scans for `.md`, `.dita`, and `.html` files.
   - Recursively searches subdirectories.
   - Loads full document content.

3. **Metadata Enhancement**:
   - Adds DITA map paths to document metadata.
   - Includes file paths and filenames for citation.

4. **Chunking**:
   - Splits documents into 128-token chunks with 20-token overlap.
   - Creates `TextNode` objects with preserved metadata.
   - Generates final chunk list for embedding.

---

### Cell 4
**Purpose**: Converts text chunks into vector embeddings and builds a searchable FAISS index.

**What it does**:
1. **Embedding Model Loading**:
   - Loads `multi-qa-distilbert-cos-v1` transformer model.
   - Optimized for question-answering tasks.

2. **Vector Generation**:
   - Embeds all chunk texts into 768-dimensional vectors.
   - Uses float32 format for FAISS compatibility.

3. **FAISS Index Creation**:
   - Builds an L2 (Euclidean distance) index.
   - Adds all vectors to the index.

4. **Persistence**:
   - Saves FAISS index (`my_faiss_index.bin`) to Google Drive.
   - Saves chunk metadata (`chunk_data.pkl`) using pickle.
   - Preserves the link between vectors and original text.

**Note**: This cell only needs to run once. After the index is saved, you can skip to Cell 5 in future sessions.

---

### Cell 5
**Purpose**: Provides an interactive interface for querying documents.

**What it does**:
1. **Component Loading**:
   - Loads the saved FAISS index from Google Drive.
   - Loads chunk metadata from pickle file.
   - Initializes the embedding model and Gemini client.

2. **Retrieval Function** (`retrieve_context`):
   - Embeds user queries into vectors.
   - Searches FAISS index for similar chunks.
   - Filters results by similarity threshold.
   - Formats context with metadata for LLM.

3. **Interactive UI**:
   - **Query Input**: Text area for queries
   - **Top K Chunks**: Controls number of retrieved chunks (1-20)
   - **Similarity Threshold**: Filters for relevance (0-50, lower = stricter)
   - **Temperature**: Controls response creativity (0-1, lower = focused, higher = creative)
   - **Max Tokens**: Limits response length (64-4096)

4. **RAG Process**:
   - Retrieves relevant context from your documents.
   - Constructs a grounded prompt with citations.
   - Sends to Gemini API for generation.
   - Displays response with source provenance.

---

## Configuration Options

### Document Processing (Cell 3)
```python
DITA_SUBFOLDER = 'Model_T_DITA'  # Change to your DITA folder name
chunk_size=128                     # Tokens per chunk
chunk_overlap=20                   # Overlap between chunks
target_extensions = ['.md', '.dita', '.html']  # Supported file types
```
**To Do**: The source directory is hardcoded, so the value of DITA_SUBFOLDER needs to be generated dynamically.

### Embedding Model (Cell 4)
```python
model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
```

### Gemini Model (Cell 5)
```python
GEMINI_MODEL = "gemini-2.5-flash"
```

## Usage Examples

### Example Query 1: Factual Question
```
Query: "Which car was named after a breed of horse?"
Top K: 5
Threshold: 10.0
Temperature: 0.2
```

### Example Query 2: Complex Analysis
```
Query: "Compare the engine specifications across different Model T variants"
Top K: 10
Threshold: 15.0
Temperature: 0.4
```

## Troubleshooting

### "No module named 'faiss'" Error
- **Cause**: Cell 1A not run or failed to install.
- **Solution**: Run Cell 1A, wait for completion, then run Cell 1B.

### "GEMINI_API_KEY not found" Error
- **Cause**: API key not set in Colab Secrets.
- **Solution**: Follow Step 3 in setup, ensure "Notebook access" is enabled.

### "Could not load FAISS index" Error
- **Cause**: Cell 4 hasn't been run yet or index path is incorrect.
- **Solution**: Run Cell 4 first to create the index.

### "No chunks met the similarity threshold" Warning
- **Cause**: Threshold too strict for the query.
- **Solution**: Increase the similarity threshold slider or try a different query.

### Empty Response from Gemini
- **Cause**: Safety filters triggered or context too large.
- **Solution**: Reduce Top K chunks or rephrase query.


## main Components

- [LlamaIndex](https://www.llamaindex.ai/) for document processing
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook AI Research
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Google Gemini](https://deepmind.google/technologies/gemini/) API

**Note**: The RAG pipeline was Tested On Google Colab (Python 3.10+).

---


Thank you Keith Schengili-Roberts for providing the [Model T Manual transformation to DITA](https://github.com/DITAWriter/Model_T_Manual_AI_DITA_Conversion/).