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

- Google account with access to Google Colab
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Source documents stored in Google Drive (DITA, Markdown, or HTML format)

## Installation & Setup

### Step 1: Clone or Download the Repository

**Option A: Clone via Git**
```bash
git clone https://github.com/shealy2020/colab-gdrive-rag-ui-1
```

**Option B: Download ZIP**
1. Click the green "Code" button on the GitHub repository page.
2. Select "Download ZIP".
3. Extract the ZIP file to your local machine.

### Step 2: Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File** → **Upload notebook**.
3. Select the `colab-gdrive-rag-ui-1.py` file from the repository.
4. Alternatively, open a new notebook and copy-paste each cell from the script.

### Step 3: Configure Your Gemini API Key

1. In Google Colab, click the **key icon** in the left sidebar (Secrets).
2. Click **Add new secret**.
3. Set the name as: `GEMINI_API_KEY`.
4. Paste your Gemini API key as the value.
5. Select **Notebook access**.

### Step 4: Prepare Your Documents

1. Create a folder structure in your Google Drive similar to this:
   ```
   MyDrive/
   └── gemini-source-index/
       ├── rag_docs_structured/          # Place your documents here
       │   └── Model_T_DITA/              # Example DITA folder
       │       ├── *.ditamap files
       │       └── *.dita files
       └── rag_index_gemini_faiss/        # Index will be saved here (auto-created)
   ```

2. Update the paths in **Cell 2** if you use different folder names:
   ```python
   DOCS_DIR = '/content/drive/MyDrive/gemini-source-index/rag_docs_structured'
   FAISS_INDEX_PATH = '/content/drive/MyDrive/gemini-source-index/rag_index_gemini_faiss'
   ```

### Step 5: Run the Notebook

Execute cells in order:
1. **Cell 1A**: Installs dependencies (wait for completion).
2. **Cell 1B**: Imports libraries and validate API key.
3. **Cell 2**: Mounts Google Drive and configure paths.
4. **Cell 3**: Loads and chunk documents.
5. **Cell 4**: Generates embeddings and build FAISS index.
6. **Cell 5**: Launches interactive query interface.

## Cell-by-Cell Breakdown

### Cell 1A: Install Dependencies
**Purpose**: Installs all required Python packages before importing them.

**What it does**:
- Installs FAISS (Facebook AI Similarity Search) for vector indexing.
- Installs Sentence Transformers for text embeddings.
- Installs Google Generative AI SDK (Gemini API client).
- Installs LlamaIndex for document processing.
- Installs supporting libraries (numpy, lxml, ipywidgets).

**Note**: Must run first before any imports in Cell 1B.

---

### Cell 1B: Setup and Import Libraries
**Purpose**: Imports all installed packages and validates your Gemini API key.

**What it does**:
- Imports core libraries (FAISS, transformers, Gemini client).
- Retrieves your `GEMINI_API_KEY` from Colab Secrets.
- Validates the API key is accessible.
- Sets up the environment for subsequent cells.

---

### Cell 2: Google Drive and Environment Setup
**Purpose**: Connects to your Google Drive and creates necessary directories or verifies that directories exist.

**What it does**:
- Mounts your Google Drive to the Colab runtime.
- Defines paths:
  - `DOCS_DIR`: Where your source documents are stored.
  - `FAISS_INDEX_PATH`: Where the vector index will be saved.
- Creates directories if they don't exist.
- Validates the file structure.

---

### Cell 3: LlamaIndex Structure-Aware Document Loading and Chunking
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

### Cell 4: Embeddings and FAISS Indexing
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

### Cell 5: User Queries, Retrieval, and Gemini RAG Response Generation
**Purpose**: Provides an interactive interface for querying your documents with AI-powered responses.

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
   - **Query Input**: Text area for your question
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