# RAG PDF Processing Pipeline

A robust pipeline for extracting structured content from PDFs for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Phase 1**: Adobe PDF extraction with intelligent formatting
- **Phase 2**: OCR processing for tables/figures using Vertex AI (Gemini 2.5 Flash)
- **Text merging** across pages under same headings
- **Coordinate preservation** for PDF viewer citation
- **Multiprocessing** for batch processing

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
# Copy example config
cp config/env.example config/.env

# Edit with your credentials
nano config/.env
```

**Required credentials:**
- **Adobe PDF Services**: Get from [Adobe Developer Console](https://developer.adobe.com/document-services/apis/pdf-services/)
- **Google Cloud/Vertex AI**: Enable Vertex AI API in [GCP Console](https://console.cloud.google.com/)

### 3. Process a Single PDF

```bash
python run_single.py "path/to/your.pdf"
```

Output will be saved to `output/` folder by default.

**With custom output directory:**
```bash
python run_single.py "path/to/your.pdf" my_output_folder
```

**Example:**
```bash
python run_single.py "/Users/john/documents/my_guideline.pdf"
```

### 4. Batch Processing

Edit `src/phase2_multiprocessing_WORKING.py` config section:

```python
config = {
    'raw_json_dir': 'path/to/raw_json_files',
    'pdf_dir': 'path/to/pdf_files',
    'output_dir': 'path/to/output',
    'num_workers': 4
}
```

Then run:
```bash
python src/phase2_multiprocessing_WORKING.py
```

## Output Format

Each processed PDF produces a JSON file with entries like:

```json
{
    "content": "DOCUMENT: filename | H1: Section | PAGE: 0 | TYPE: text || Content here...",
    "meta": {
        "title": "filename",
        "coordinate": [x1, y1, x2, y2],
        "Doc_type": "text",
        "headings": {"1": "Section Title"},
        "page": 0
    }
}
```

## File Structure

```
rag_pipeline_clean/
├── run_single.py                     # Process single PDF (command line)
├── run_batch.py                      # Batch processing (command line)
├── src/
│   ├── pdf_processor_organized.py    # Main processor (Phase 1 + 2)
│   └── phase2_multiprocessing_WORKING.py  # Multiprocessing wrapper
├── config/
│   ├── .env                          # Your credentials (not in git)
│   ├── env.example                   # Credential template
│   └── pdfservices-api-credentials.json.example
├── output/                           # Processed JSON files
├── requirements.txt
└── README.md
```

## Pipeline Flow

```
PDF File
    │
    ▼
┌─────────────────────────────────┐
│  Phase 1: Adobe PDF Extract     │
│  - Extract raw JSON             │
│  - Cache results                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  JSON Formatting                │
│  - Parse headings (H1, H2...)   │
│  - Skip boilerplate text        │
│  - Preserve coordinates         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 2: OCR Processing        │
│  - Capture table/figure images  │
│  - Send to Gemini 2.5 Flash     │
│  - Add summaries + confidence   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Post-Processing                │
│  - Merge text across pages      │
│  - Merge consecutive tables     │
│  - Final JSON output            │
└─────────────────────────────────┘
    │
    ▼
Processed JSON (for RAG)
```

## Notes

- The pipeline filters Monash Health-specific boilerplate (PROMPT Doc No, etc.)
- Adjust skip patterns in `pdf_processor_organized.py` for other document types
- OCR uses Vertex AI Gemini 2.5 Flash - ensure quota is available

