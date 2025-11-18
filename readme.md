# RAG System with Google Colab, FAISS, and Gemini API

A comprehensive Retrieval-Augmented Generation (RAG) system that processes structured documentation (including DITA maps), creates vector embeddings, and provides an interactive query interface powered by Google's Gemini AI.

## Features

- **Structure-Aware Document Processing**: Supports DITA maps, Markdown, HTML, and preserves document hierarchy
- **FAISS Vector Search**: Fast similarity search using Facebook's FAISS library
- **Gemini AI Integration**: Leverages Google's Gemini 2.5 Flash model for intelligent responses
- **Interactive UI**: Built-in Jupyter widgets for easy query testing
- **Persistent Storage**: Saves indexes to Google Drive for reuse across sessions
- **Configurable Retrieval**: Adjustable similarity thresholds and top-K results

## Prerequisites

- Google account with access to Google Colab
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Documents stored in Google Drive (DITA, Markdown, or HTML format)

## Installation & Setup

### Step 1: Clone or Download the Repository

**Option A: Clone via Git**
```bash
git clone https://github.com/yourusername/your-repo-name.git
```

**Option B: Download ZIP**
1. Click the green "Code" button on the GitHub repository page
2. Select "Download ZIP"
3. Extract the ZIP file to your local machine

### Step 2: Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Select the `colab-gdrive-rag-ui-1.py` file from the repository
4. Alternatively, open a new notebook and copy-paste each cell from the script

### Step 3: Configure Your Gemini API Key

1. In Google Colab, click the **🔑 key icon** in the left sidebar (Secrets)
2. Click **Add new secret**
3. Set the name as: `GEMINI_API_KEY`
4. Paste your Gemini API key as the value
5. Toggle on **Notebook access**

### Step 4: Prepare Your Documents

1. Create a folder structure in your Google Drive:
   ```
   MyDrive/
   └── gemini-api-8/
       ├── rag_docs_structured/          # Place your documents here
       │   └── Model_T_DITA/              # Example DITA folder
       │       ├── *.ditamap files
       │       └── *.dita files
       └── rag_index_gemini_faiss/        # Index will be saved here (auto-created)
   ```

2. Update the paths in **Cell 2** if you use different folder names:
   ```python
   DOCS_DIR = '/content/drive/MyDrive/gemini-api-8/rag_docs_structured'
   FAISS_INDEX_PATH = '/content/drive/MyDrive/gemini-api-8/rag_index_gemini_faiss'
   ```

### Step 5: Run the Notebook

Execute cells in order:
1. **Cell 1A**: Install dependencies (wait for completion)
2. **Cell 1B**: Import libraries and validate API key
3. **Cell 2**: Mount Google Drive and configure paths
4. **Cell 3**: Load and chunk documents
5. **Cell 4**: Generate embeddings and build FAISS index
6. **Cell 5**: Launch interactive query interface

## Cell-by-Cell Breakdown

### Cell 1A: Install Dependencies
**Purpose**: Installs all required Python packages before importing them.

**What it does**:
- Installs FAISS (Facebook AI Similarity Search) for vector indexing
- Installs Sentence Transformers for text embeddings
- Installs Google Generative AI SDK (Gemini API client)
- Installs LlamaIndex for document processing
- Installs supporting libraries (numpy, lxml, ipywidgets)

**Runtime**: ~30-60 seconds

**Note**: Must run first before any imports!

---

### Cell 1B: Setup and Import Libraries
**Purpose**: Imports all installed packages and validates your Gemini API key.

**What it does**:
- Imports core libraries (FAISS, transformers, Gemini client)
- Retrieves your `GEMINI_API_KEY` from Colab Secrets
- Validates the API key is present and accessible
- Sets up the environment for subsequent cells

**Expected Output**: "Gemini API Key successfully loaded and all imports completed."

**Troubleshooting**: If you see an error about missing API key, verify Step 3 above.

---

### Cell 2: Google Drive and Environment Setup
**Purpose**: Connects to your Google Drive and creates necessary directories.

**What it does**:
- Mounts your Google Drive to the Colab runtime
- Defines paths for:
  - `DOCS_DIR`: Where your source documents are stored
  - `FAISS_INDEX_PATH`: Where the vector index will be saved
- Creates directories if they don't exist
- Validates the file structure

**Expected Output**: "Google Drive mounted and environment paths set."

**User Action Required**: Authorize Google Drive access when prompted.

---

### Cell 3: LlamaIndex Structure-Aware Document Loading and Chunking
**Purpose**: Loads your documents, parses DITA map structure, and splits content into chunks.

**What it does**:
1. **DITA Map Parsing**: 
   - Recursively parses `.ditamap` files to extract document hierarchy
   - Builds a path map (e.g., "Manual > Chapter 1 > Topic Name")
   - Preserves structural context for better retrieval

2. **Document Loading**:
   - Scans for `.md`, `.dita`, and `.html` files
   - Recursively searches subdirectories
   - Loads full document content

3. **Metadata Enhancement**:
   - Adds DITA map paths to document metadata
   - Includes file paths and filenames for citation

4. **Chunking**:
   - Splits documents into 128-token chunks with 20-token overlap
   - Creates `TextNode` objects with preserved metadata
   - Generates final chunk list for embedding

**Expected Output**: 
- "Found X .ditamap file(s)"
- "Loaded X source document(s)"
- "Successfully split content into X final, structured chunks"

**Key Variables Created**:
- `chunks`: List of dictionaries containing text and metadata
- `DITA_PATH_MAP`: Mapping of DITA files to their structural paths

---

### Cell 4: Embeddings and FAISS Indexing
**Purpose**: Converts text chunks into vector embeddings and builds a searchable FAISS index.

**What it does**:
1. **Embedding Model Loading**:
   - Loads `multi-qa-distilbert-cos-v1` model
   - Optimized for question-answering tasks

2. **Vector Generation**:
   - Embeds all chunk texts into 768-dimensional vectors
   - Uses float32 format for FAISS compatibility

3. **FAISS Index Creation**:
   - Builds an L2 (Euclidean distance) index
   - Adds all vectors to the index

4. **Persistence**:
   - Saves FAISS index (`my_faiss_index.bin`) to Google Drive
   - Saves chunk metadata (`chunk_data.pkl`) using pickle
   - Preserves the link between vectors and original text

**Expected Output**: 
- "Embedding model loaded: sentence-transformers/multi-qa-distilbert-cos-v1"
- "Generating embeddings for X chunks..."
- "FAISS index built successfully with dimension: 768"
- "FAISS index (Vectors) saved to: [path]"
- "Chunk metadata saved to: [path]"

**Runtime**: Varies based on document count (1-5 minutes for ~1000 chunks)

**Important**: This cell only needs to run once. After the index is saved, you can skip to Cell 5 in future sessions.

---

### Cell 5: User Queries, Retrieval, and Gemini RAG Response Generation
**Purpose**: Provides an interactive interface for querying your documents with AI-powered responses.

**What it does**:
1. **Component Loading**:
   - Loads the saved FAISS index from Google Drive
   - Loads chunk metadata from pickle file
   - Initializes the embedding model and Gemini client

2. **Retrieval Function** (`retrieve_context`):
   - Embeds user queries into vectors
   - Searches FAISS index for similar chunks
   - Filters results by similarity threshold
   - Formats context with metadata for LLM

3. **Interactive UI**:
   - **Query Input**: Text area for your question
   - **Top K Chunks**: Slider to control number of retrieved chunks (1-20)
   - **Similarity Threshold**: Filter for relevance (0-50, lower = stricter)
   - **Temperature**: Controls response creativity (0-1, lower = focused)
   - **Max Tokens**: Limits response length (64-4096)

4. **RAG Process**:
   - Retrieves relevant context from your documents
   - Constructs a grounded prompt with citations
   - Sends to Gemini API for generation
   - Displays response with source provenance

**Expected Output**: 
- Interactive widget interface
- Query results with:
  - Gemini's response
  - Source citations with DITA map paths
  - L2 distance scores
  - Content snippets from retrieved chunks

**Key Features**:
- Adjustable retrieval parameters in real-time
- Source attribution for every claim
- Safety filter handling and error messages
- Distance-based relevance scoring

---

## Configuration Options

### Document Processing (Cell 3)
```python
DITA_SUBFOLDER = 'Model_T_DITA'  # Change to your DITA folder name
chunk_size=128                     # Tokens per chunk
chunk_overlap=20                   # Overlap between chunks
target_extensions = ['.md', '.dita', '.html']  # Supported file types
```

### Embedding Model (Cell 4)
```python
model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
# Alternatives:
# - "sentence-transformers/all-MiniLM-L6-v2" (faster, smaller)
# - "sentence-transformers/all-mpnet-base-v2" (more accurate)
```

### Gemini Model (Cell 5)
```python
GEMINI_MODEL = "gemini-2.5-flash"
# Alternative: "gemini-1.5-pro" (more capable, slower)
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
- **Cause**: Cell 1A not run or failed to install
- **Solution**: Run Cell 1A, wait for completion, then run Cell 1B

### "GEMINI_API_KEY not found" Error
- **Cause**: API key not set in Colab Secrets
- **Solution**: Follow Step 3 in setup, ensure "Notebook access" is enabled

### "Could not load FAISS index" Error
- **Cause**: Cell 4 hasn't been run yet or index path is incorrect
- **Solution**: Run Cell 4 first to create the index

### "No chunks met the similarity threshold" Warning
- **Cause**: Threshold too strict for the query
- **Solution**: Increase the similarity threshold slider or try a different query

### Empty Response from Gemini
- **Cause**: Safety filters triggered or context too large
- **Solution**: Reduce Top K chunks or rephrase query

## Performance Tips

1. **First Run**: Cells 1-4 take ~5-10 minutes total
2. **Subsequent Runs**: Only run Cell 5 if index already exists (30 seconds)
3. **Large Document Sets**: Consider increasing chunk_size to reduce total chunks
4. **Memory Issues**: Reduce Top K or use a smaller embedding model

## File Structure

```
your-repo/
├── README.md                           # This file
├── colab-gdrive-rag-ui-1.py           # Main Colab notebook script
└── example_docs/                       # (Optional) Sample documents
    └── example.md
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

[Your chosen license - e.g., MIT, Apache 2.0]

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for document processing
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook AI Research
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Google Gemini](https://deepmind.google/technologies/gemini/) API

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the [Gemini API documentation](https://ai.google.dev/docs)

---

**Last Updated**: November 2025
**Tested On**: Google Colab (Python 3.10+)