# Cell 1A: Install Dependencies FIRST

# Install all required packages before importing
import pickle  # Used to save Python objects like the list of chunk dictionaries
# Standard library for XML parsing (DITA maps)
import xml.etree.ElementTree as ET
from google.colab import drive
import os
from google.colab import drive, userdata
import re
from typing import List, Dict, Callable
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import xml.etree.ElementTree as ET
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from google import genai
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
!pip install - q faiss-cpu sentence-transformers google-genai numpy llama-index lxml ipywidgets

print("All dependencies installed successfully. Please proceed to Cell 1B.")


# Cell 1B: Setup and Import Libraries

# Standard library for XML parsing (DITA maps)

# --- API Key Retrieval from Colab Secrets ---
# This ensures the official 'google-genai' SDK is authenticated
try:
    os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
except Exception as e:
    raise ValueError("GEMINI_API_KEY not found or accessible. Please ensure it's set in Colab Secrets (the '🔑' icon on the left panel) and run this cell interactively in the Colab UI.") from e

if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
    raise ValueError("GEMINI_API_KEY not found in Colab Secrets.")
print("Gemini API Key successfully loaded and all imports completed.")

# Cell 2: Google Drive and Environment Setup


# Mount Google Drive to access documents and save the index
drive.mount('/content/drive')

# --- Configuration (UPDATED PATHS) ---
# Define the path where your source documents are located on Google Drive
DOCS_DIR = '/content/drive/MyDrive/gemini-api-8/rag_docs_structured'
# Define the path where the FAISS index will be saved.
FAISS_INDEX_PATH = '/content/drive/MyDrive/gemini-api-8/rag_index_gemini_faiss'
# Filename for the raw FAISS binary index
FAISS_INDEX_FILE = 'my_faiss_index.bin'

# Check/Create the document directory
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    print(f"Created document directory: {DOCS_DIR}")
else:
    print(f"Document directory confirmed: {DOCS_DIR}")


# Create the index parent directory if it doesn't exist
if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH)
    print(f"Created directory for index: {FAISS_INDEX_PATH}")
else:
    print(f"Index directory confirmed: {FAISS_INDEX_PATH}")

print("\nGoogle Drive mounted and environment paths set.")


# Cell 3: LlamaIndex Structure-Aware Document Loading and Chunking (DITA Map Integration with Fixed Metadata Handling)


# DOCS_DIR is defined in Cell 2

# Global map to manage state across map files
# Key: topic filename (e.g., 'concept.dita'), Value: DITA map path (e.g., 'Map Title > Chapter 1 > Topic')
DITA_PATH_MAP: Dict[str, str] = {}
chunks: List[Dict] = []  # Stores the final, processed chunks with metadata


def parse_ditamap(ditamap_path: str, current_path: List[str] = []) -> None:
    """
    Recursively parses the DITA map, building the structural path for all referenced topics.

    This simplified version handles only direct @href references, ignoring keydefs.
    """

    ET.register_namespace(
        '', 'http://dita.oasis-open.org/spec/DITA/topics/dita-map')

    try:
        tree = ET.parse(ditamap_path)
    except ET.ParseError as e:
        print(f"Error parsing DITA map {os.path.basename(ditamap_path)}: {e}")
        return

    root = tree.getroot()

    # 1. Initialize path and extract map title
    if not current_path and root.tag.endswith('bookmap'):
        # Attempt to get the main book title
        booktitle_element = root.find('.//{*}booktitle/{*}mainbooktitle')
        map_title = (booktitle_element.text.strip() if booktitle_element is not None and booktitle_element.text else
                     os.path.basename(ditamap_path).replace('.ditamap', ''))
        current_path.append(map_title)
    elif not current_path and root.tag.endswith('map'):
        map_title = root.get('title') or os.path.basename(
            ditamap_path).replace('.ditamap', '')
        current_path.append(map_title)

    # 2. Path Traversal Pass: Traverse the map structure
    def recurse_topicrefs(element: ET.Element, path: List[str]):
        """Helper to traverse chapters/topicrefs and build the path."""
        for child in element:
            # Check for chapter or topicref elements
            if child.tag.endswith(('chapter', 'topicref')):

                # Use navtitle, then title, then fallback to filename (if no title)
                title = child.get('navtitle') or child.get('title')
                target_uri = child.get('href')  # Direct reference assumed

                if target_uri:
                    new_path = path[:]
                    # Only add the title to the path if it's explicitly available
                    if title:
                        new_path.append(title)
                    # If no explicit title, use the filename as the last segment for clarity
                    elif not title and target_uri.endswith('.dita'):
                        new_path.append(os.path.basename(
                            target_uri).replace('.dita', ''))

                    filename = os.path.basename(target_uri)

                    if filename.endswith('.dita'):
                        full_path_str = " > ".join(new_path)
                        # Store the final file path with its structural context
                        DITA_PATH_MAP[filename] = full_path_str

                    elif filename.endswith(('.ditamap', '.map')):
                        # Recursively process nested maps, passing the current path state
                        nested_map_path = os.path.join(
                            os.path.dirname(ditamap_path), target_uri)
                        parse_ditamap(nested_map_path, new_path)

                # Recurse for nested topicrefs/chapters using the updated path
                recurse_topicrefs(child, new_path)

    recurse_topicrefs(root, current_path)
    # print(f"  Processed structure for '{os.path.basename(ditamap_path)}'.") # Removed for conciseness


print("--- 1. Preparing Source Documents (LlamaIndex & DITA Map Integration - FIX APPLIED) ---")

# Based on user feedback, the DITA content is nested within a subdirectory.
DITA_SUBFOLDER = 'Model_T_DITA'
DITA_MAP_ROOT_DIR = os.path.join(DOCS_DIR, DITA_SUBFOLDER)

# 1A. Step 1: Re-Parse the DITA Maps to build the structural path map
# Ensure DITA_PATH_MAP is clear before repopulating to avoid stale data
global DITA_PATH_MAP
DITA_PATH_MAP = {}

print(f"Searching for .ditamap files inside: {DITA_MAP_ROOT_DIR}")
ditamap_files = []

# Search specifically within the DITA map root directory
for root, _, files in os.walk(DITA_MAP_ROOT_DIR):
    for file in files:
        if file.endswith('.ditamap'):
            ditamap_files.append(os.path.join(root, file))

if ditamap_files:
    print(f"Found {len(ditamap_files)} .ditamap file(s).")

    for map_file in ditamap_files:
        parse_ditamap(map_file)

    print(
        f"Finished parsing DITA map structure. Found and mapped {len(DITA_PATH_MAP)} unique DITA topics.")
else:
    print("No .ditamap files found. Proceeding with standard file loading.")


# 1B. Step 2: Load Documents using LlamaIndex Reader (WITHOUT file_metadata argument)
target_extensions = ['.md', '.dita', '.html']
loader = SimpleDirectoryReader(
    input_dir=DOCS_DIR,  # Search starts from the top DOCS_DIR
    recursive=True,  # Critical for finding files in subdirectories like Model_T_Manual_AI_DITA_Conversion-main/topics
    required_exts=target_extensions
)
documents: List[Document] = loader.load_data()
print(f"Loaded {len(documents)} source document(s) from content files.")

# Two-step Meta
# --- FIX: Post-process documents to add DITA map paths ---
print("Applying DITA map paths to documents...")
for doc in documents:
    # Ensure 'file_path' exists in metadata before proceeding
    if 'file_path' in doc.metadata:
        basename = os.path.basename(doc.metadata['file_path'])
        if basename.endswith('.dita') and basename in DITA_PATH_MAP:
            doc.metadata['dita_map_path'] = DITA_PATH_MAP[basename]
print("DITA map paths applied.")
# --- END FIX ---


# 2. Configure a Structure-Aware Node Parser (Chunker)
parser = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=20
)

# 3. Get Nodes (Chunks)
nodes: List[TextNode] = parser.get_nodes_from_documents(documents)


# 4. Convert LlamaIndex Nodes back to the required 'chunks' dictionary format
chunks = []  # Clear and re-populate the global 'chunks' variable
for node in nodes:
    # LlamaIndex metadata includes 'file_path' and now potentially 'dita_map_path'
    metadata = node.metadata

    # Extract the filename for citation purposes
    metadata['filename'] = os.path.basename(metadata.get('file_path', 'N/A'))

    chunks.append({
        "text": node.text,
        "metadata": metadata
    })

print(
    f"\nSuccessfully split content into {len(chunks)} final, structured chunks.")

# Example inspection - removed verbose checks for specific files as per request
if chunks:
    print("\nExample Structured Chunk 1:")
    print(f"  Content: '{chunks[0]['text'][:100]}...' ")
    print(f"  Metadata: {chunks[0]['metadata']}")
else:
    print("Warning: No chunks were created.")

# Cell 4: Embeddings and FAISS Indexing


# Paths and variables defined in Cell 2 and 3
# chunks: list of dicts with 'text' and 'metadata'
# FAISS_INDEX_PATH and FAISS_INDEX_FILE are defined in Cell 2

print("--- 2. Creating Embeddings and Vectors ---")

# 1. Load the Sentence Transformer Model
# Updated embedding model
model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
model = SentenceTransformer(model_name)
print(f"Embedding model loaded: {model_name}")

# 2. Prepare Chunk Text for Embedding
chunk_texts = [chunk['text'] for chunk in chunks]

if not chunk_texts:
    raise ValueError(
        "No text chunks found. Please check Cell 3 output and document loading.")

# 3. Generate Embeddings
print(f"Generating embeddings for {len(chunk_texts)} chunks...")
embeddings = model.encode(chunk_texts, convert_to_numpy=True).astype('float32')

# 4. Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index built successfully with dimension: {dimension}")

# --- 3. Indexing and Metadata (Persistence) ---
print("\n--- 3. Indexing and Metadata (Persistence) ---")

# Save the raw FAISS index
full_index_path = os.path.join(FAISS_INDEX_PATH, FAISS_INDEX_FILE)
faiss.write_index(index, full_index_path)
print(f"FAISS index (Vectors) saved to: {full_index_path}")

# FIX: Save the entire original 'chunks' list using pickle.
# This preserves the link between the FAISS vector index (row numbers) and the chunk text/metadata.
chunk_data_file = os.path.join(FAISS_INDEX_PATH, 'chunk_data.pkl')
with open(chunk_data_file, 'wb') as f:
    pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Chunk metadata saved to: {chunk_data_file}")


# Cell 5: User Queries, Retrieval, and Gemini RAG Response Generation (Interactive UI with Similarity Threshold)


# --- Configuration and Initialization ---
# Paths and variables defined in Cell 2
FAISS_INDEX_PATH = '/content/drive/MyDrive/gemini-api-8/rag_index_gemini_faiss'
FAISS_INDEX_FILE = 'my_faiss_index.bin'
full_index_path = os.path.join(FAISS_INDEX_PATH, FAISS_INDEX_FILE)
chunk_data_file = os.path.join(FAISS_INDEX_PATH, 'chunk_data.pkl')
model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
GEMINI_MODEL = "gemini-2.5-flash"

# 1. Load Components
print("--- 4. User Queries & Retrieval Setup (Loading Components) ---")

# Load the FAISS Index
try:
    index = faiss.read_index(full_index_path)
    print(f"FAISS index loaded successfully from {full_index_path}")
except Exception as e:
    raise FileNotFoundError(
        f"Could not load FAISS index: {e}. Ensure Cell 4 ran correctly.")

# Load the full chunk data (text and metadata) using pickle
try:
    with open(chunk_data_file, 'rb') as f:
        full_chunk_data = pickle.load(f)
    print(f"Full chunk data loaded for {len(full_chunk_data)} chunks.")
except Exception as e:
    raise FileNotFoundError(
        f"Could not load chunk data: {e}. Ensure Cell 4 ran correctly.")

# Load the Embedding Model
model = SentenceTransformer(model_name)
print(f"Embedding model loaded: {model_name}")

# Initialize the Gemini Client
client = genai.Client()


# --- 2. Retrieval Function with Similarity Threshold ---
def retrieve_context(query: str, k: int = 5, similarity_threshold: float = None) -> (str, list):
    """Embeds the query, searches the FAISS index, and formats the context.

    Args:
        query: The user's question
        k: Maximum number of chunks to retrieve
        similarity_threshold: Maximum L2 distance to include (lower is more similar).
                            If None, all k chunks are returned.

    Returns:
        tuple: (full_context_string, list_of_retrieved_chunks_with_distances)
    """
    if not query:
        return "", []

    # 1. Embed the query
    query_vector = model.encode(
        [query], convert_to_numpy=True).astype('float32')

    # 2. Search the FAISS index
    D, I = index.search(query_vector, k)

    # 3. Extract the original chunk data using the indices
    retrieved_chunks = []

    for i, idx in enumerate(I[0]):
        distance = D[0][i]

        # Apply similarity threshold filter if specified
        if similarity_threshold is not None and distance > similarity_threshold:
            continue  # Skip chunks that don't meet the similarity threshold

        original_chunk = full_chunk_data[idx]
        metadata = original_chunk['metadata']

        # Incorporate DITA Map Structural Path (if available)
        metadata_str = ""
        dita_path = metadata.get('dita_map_path')
        file_path = metadata.get('file_path')

        if dita_path:
            metadata_str += f"Document Structure: {dita_path}\n"

        if file_path:
            metadata_str += f"Document Path: {file_path}\n"

        # Access 'text' and 'filename' correctly from the loaded dictionary
        context_text = (
            f"Source File: {metadata.get('filename', 'N/A')}\n"
            f"{metadata_str}"
            f"Content: {original_chunk['text']}\n---\n"
        )

        retrieved_chunks.append({
            "text": context_text,
            "metadata": metadata,
            "distance": distance
        })

    # Concatenate the text into a single context string for the LLM
    full_context = "".join([c["text"] for c in retrieved_chunks])

    return full_context, retrieved_chunks


# --- 3. Interactive UI Definition and Execution ---

# Define the widgets
query_widget = widgets.Textarea(
    value='',
    description='Query:',
    placeholder='Which car was named after a breed of horse?',
    layout=widgets.Layout(width='auto', height='80px')
)

top_k_widget = widgets.IntSlider(
    value=5,
    min=1,
    max=20,
    step=1,
    description='Top K Chunks:',
    style={'description_width': 'initial'},
    tooltip='Number of document chunks to retrieve'
)

similarity_threshold_widget = widgets.FloatSlider(
    value=10.0,
    min=0.0,
    max=50.0,
    step=0.5,
    description='Similarity Threshold:',
    style={'description_width': 'initial'},
    tooltip='Max distance filter; lower means stricter'
)

temperature_widget = widgets.FloatSlider(
    value=0.2,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Temperature:',
    style={'description_width': 'initial'},
    tooltip='Controls response randomness; lower is focused'
)

max_tokens_widget = widgets.IntSlider(
    value=1024,
    min=64,
    max=4096,
    step=64,
    description='Max Tokens:',
    style={'description_width': 'initial'},
    tooltip='Maximum length of generated response'
)

submit_button = widgets.Button(
    description='Run RAG Query',
    button_style='success',
    icon='play'
)

output_area = widgets.Output()

# Assemble the UI layout
input_form = widgets.VBox([
    query_widget,
    widgets.HBox([
        top_k_widget,
        similarity_threshold_widget
    ]),
    widgets.HBox([
        temperature_widget,
        max_tokens_widget
    ]),
    submit_button
], layout=widgets.Layout(border='1px solid lightgray', padding='10px', width='100%'))

# Define the function to run on button click


def on_button_click(b):
    with output_area:
        clear_output(wait=True)
        query = query_widget.value.strip()
        k = top_k_widget.value
        similarity_threshold = similarity_threshold_widget.value if similarity_threshold_widget.value < 50.0 else None
        temp = temperature_widget.value
        max_t = max_tokens_widget.value

        if not query:
            print("Please enter a query to run the RAG process.")
            return

        threshold_str = f"{similarity_threshold:.2f}" if similarity_threshold is not None else "disabled"
        print(
            f"Retrieving context for query: **{query}** (k={k}, threshold={threshold_str})")

        # 1. Retrieval
        context, source_docs = retrieve_context(
            query, k=k, similarity_threshold=similarity_threshold)

        if not source_docs:
            print("\n⚠️ No chunks met the similarity threshold. Try increasing the threshold or lowering the similarity requirement.")
            return

        print(
            f"Retrieved {len(source_docs)} chunks that met the similarity criteria.")

        # 2. Construct the Final Prompt for grounding
        system_prompt = f"""Use ONLY the following pieces of context to answer the user's question.
Your answer MUST be based ONLY on the provided context. Do not use external knowledge or make up facts.
For every piece of information used, cite the source filename (from the 'Source File:' line in the context).
If a 'Document Structure:' is provided, reference that path in your citation to show context.

Context:
{context}

Question: {query}
Helpful Answer:"""

        # 3. Call the Gemini API
        print("Sending prompt to Gemini...")
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=system_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_t
                )
            )

            # --- TROUBLESHOOTING FIX START ---
            if not response.candidates:
                # The model returned no response candidate (often due to safety block or internal error)
                feedback = response.prompt_feedback

                print("\n" + "="*50)
                display(
                    Markdown("## ❌ Gemini API Error: No Response Candidate Found"))
                print(f"**Reason:** The model returned an empty response object.")

                if feedback and feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                    block_reason_str = feedback.block_reason.name
                    safety_ratings = [
                        f"{r.category.name} ({r.probability.name})" for r in feedback.safety_ratings]

                    display(Markdown(f"### 🛑 Blocked By Safety Policy"))
                    print(f"Block Reason: **{block_reason_str}**")
                    print("Safety Ratings that caused the block:")
                    for rating in safety_ratings:
                        print(f"- {rating}")
                    print("\n**Suggestion:** The prompt (Query + Retrieved Context) triggered a block. Try lowering **Top K Chunks** to reduce the context or rephrase your query.")
                else:
                    # Generic failure, e.g., service error, timeout, or context size issue
                    print("Could not determine specific block reason. This might be a transient API issue or the total input context size is too large for the model.")

            elif not response.text:
                # This case is when candidates exist but text is empty (rare, but possible with safety block)
                display(Markdown("## ⚠️ Response Warning: Empty Text Returned"))
                print("The API returned an empty text string despite having candidates. Retrying or slightly modifying the prompt may help.")

            # --- Original successful print logic ---
            else:
                # --- Print Results ---
                print("\n" + "="*50)
                display(Markdown("## ✅ Final Gemini RAG Response:"))
                display(Markdown(response.text.strip()))

                print("\n--- Provenance (Source Documents Used) ---")
                for i, doc in enumerate(source_docs):
                    # Prioritize dita_map_path for provenance display
                    path = doc['metadata'].get('dita_map_path', doc['metadata'].get(
                        'file_path', doc['metadata'].get('filename', 'N/A')))
                    display(Markdown(
                        f"* **Chunk {i+1}** Source Location: `{path}`  (L2 Distance: {doc['distance']:.4f})"))
                    # Display the first 100 characters of the content used
                    print(
                        f"  Snippet: {doc['text'].split('Content:')[1].strip()[:100].replace('\n', ' ')}...")
                print("="*50)

            # --- TROUBLESHOOTING FIX END ---

        except Exception as e:
            print(f"An error occurred during API call: {e}")


# Link the button click event to the function
submit_button.on_click(on_button_click)

# Display the UI
print("\n--- Interactive RAG Interface ---")
display(input_form, output_area)
print("Please enter your query and parameters, then click 'Run Query'.")
