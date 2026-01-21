"""
Organized PDF Processing Pipeline for RAG System
===============================================

CHANGELOG (2025-07-10):
-----------------------
- Integrated robust table extraction:
    * Table detection regex now matches all Adobe table paths (including /Sect, Table[2], etc.).
    * Added logic to extract and preserve table structure (rows/cells) from raw JSON.
    * Output tables as structured objects, not just text blobs.
- Added raw JSON caching:
    * Before extracting with Adobe, check for a cached raw JSON file in raw_json_cache_10pdfs/raw_extractions.
    * If found, use cached raw JSON; if not, extract and save for future use.
    * All processed PDFs now save their raw JSON for future use.
- Motivation: Fixes critical data loss where table structure was lost, and prevents redundant Adobe API calls.

This module provides a clean, modular approach to processing PDFs for RAG systems.
It separates concerns into distinct components:

1. AdobePDFExtractor: Handles raw PDF extraction using Adobe services
2. JSONFormatter: Formats Adobe's raw JSON into structured content
3. OCRProcessor: Handles image capture and OCR processing
4. PDFProcessor: Main orchestrator class

Usage:
    processor = PDFProcessor(pdf_path)
    processor.process()  # Full pipeline
    
    # Or use individual components:
    extractor = AdobePDFExtractor()
    raw_json = extractor.extract_pdf_to_json(pdf_path)
    
    formatter = JSONFormatter()
    formatted_json = formatter.format_json(raw_json, pdf_filename)
    
    ocr_processor = OCRProcessor(pdf_path)
    ocr_processor.process_tables_and_figures(formatted_json)
"""

import os
import json
import zipfile
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env file
    for env_path in [
        Path(__file__).parent.parent / "config" / ".env",
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


# Adobe PDF Services imports
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

# Image processing imports
import fitz  # PyMuPDF
from PIL import Image
import io

# Vertex AI imports for Gemini 2.5 Flash
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Vertex AI
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "clinical-guideline-rag")
LOCATION = "us-central1"  # Vertex AI location for Gemini 2.5 Flash
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    logging.info(f"✅ Vertex AI initialized with project: {PROJECT_ID}, location: {LOCATION}")
except Exception as e:
    logging.error(f"❌ Failed to initialize Vertex AI: {e}")
    raise

# Adobe API Credentials (from environment variables)
ADOBE_CLIENT_ID = os.getenv("ADOBE_CLIENT_ID")
ADOBE_CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET")

if not ADOBE_CLIENT_ID or not ADOBE_CLIENT_SECRET:
    logging.warning("⚠️ Adobe credentials not found in environment. Set ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET.")


class AdobePDFExtractor:
    """Handles raw PDF extraction using Adobe PDF Services API."""
    
    def __init__(self, client_id: str = ADOBE_CLIENT_ID, client_secret: str = ADOBE_CLIENT_SECRET):
        self.credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
    
    def extract_pdf_to_json(self, pdf_path: str) -> Dict:
        """
        Extract raw JSON data from PDF using Adobe PDF Services.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing the raw JSON data from Adobe
            
        Raises:
            ServiceApiException, ServiceUsageException, SdkException
        """
        try:
            # Read PDF file
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()

            # Create PDF Services instance
            pdf_services = PDFServices(credentials=self.credentials)

            # Upload PDF and create extraction job
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
            extract_pdf_params = ExtractPDFParams(elements_to_extract=[ExtractElementType.TEXT])
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
            
            # Submit and get results
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Create temporary zip file
            zip_file_path = self._create_temp_zip_path(pdf_path)
            with open(zip_file_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

            # Extract JSON from zip
            raw_json = self._extract_json_from_zip(zip_file_path)
            
            # Clean up
            os.remove(zip_file_path)
            logging.info(f"Deleted temporary zip file: {zip_file_path}")
            
            return raw_json

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Adobe PDF extraction failed: {e}')
            raise

    def _create_temp_zip_path(self, pdf_path: str) -> str:
        """Generate temporary zip file path."""
        base_name = os.path.basename(pdf_path).replace('.pdf', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"temp_{base_name}_{timestamp}.zip"

    def _extract_json_from_zip(self, zip_file_path: str) -> Dict:
        """Extract JSON data from Adobe's zip response."""
        with zipfile.ZipFile(zip_file_path, 'r') as archive:
            for file_name in archive.namelist():
                if file_name.endswith(".json"):
                    with archive.open(file_name) as json_file:
                        json_data = json_file.read().decode("utf-8")
                        return json.loads(json_data)
        raise ValueError("No JSON file found in Adobe response")


class MetadataBuilder:
    """Helper class for building metadata prefixes."""
    
    @staticmethod
    def create_metadata_prefix(title: str, h1: str, h2: str, page: int, 
                             doc_type: str, subheadings: Optional[Dict[int, str]] = None) -> str:
        """
        Create a metadata prefix string with document context.
        
        Args:
            title: Document title
            h1: H1 heading
            h2: H2 heading  
            page: Page number
            doc_type: Type of content (text, table, figure)
            subheadings: Additional heading levels {level: text}
            
        Returns:
            Formatted metadata prefix string
        """
        prefix_parts = []
        
        if title:
            prefix_parts.append(f"DOCUMENT: {title}")
        
        # Initialize subheadings if None
        if subheadings is None:
            subheadings = {}
        
        # Add H1 and H2 to subheadings
        if h1:
            subheadings[1] = h1
        if h2:
            subheadings[2] = h2
        
        # Add all headings in order
        for level in sorted(subheadings.keys()):
            if subheadings[level]:
                prefix_parts.append(f"H{level}: {subheadings[level]}")
        
        if page is not None:
            prefix_parts.append(f"PAGE: {page}")
        
        if doc_type:
            prefix_parts.append(f"TYPE: {doc_type}")
        
        return " | ".join(prefix_parts) + " || " if prefix_parts else ""


class JSONFormatter:
    """Formats raw Adobe JSON into structured content for RAG processing."""
    
    def __init__(self):
        self.metadata_builder = MetadataBuilder()
        # FIXED: Updated regex to match only table roots, not individual cells/rows
        # This prevents individual table elements from being processed separately
        # FIXED FINAL: Handle ALL Sect patterns - single, numbered, nested, and direct
        # Patterns: //Document/Table, //Document/Sect/Table, //Document/Sect[4]/Table, //Document/Sect[5]/Sect[2]/Sect/Table
        self.table_path_re = re.compile(r"^//Document(?:/Sect(?:\[\d+\])?)*/?Table(?:\[\d+\])?$", re.IGNORECASE)
        # FIXED FINAL: Updated figure regex to handle ALL Sect patterns - single, numbered, nested, and direct
        # Patterns: //Document/Figure, //Document/Sect/Figure, //Document/Sect[4]/Figure, //Document/Sect[5]/Sect[2]/Sect/Figure
        self.figure_pattern = re.compile(r"^//Document(?:/Sect(?:\[\d+\])?)*/?Figure(?:\[\d+\])?$", re.IGNORECASE)
        self.heading_pattern = re.compile(r'h(\d+)', re.IGNORECASE)

    def format_json(self, raw_json: Dict, pdf_filename: str) -> List[Dict]:
        """
        Convert raw Adobe JSON into structured format for RAG processing.
        
        Args:
            raw_json: Raw JSON from Adobe PDF Services
            pdf_filename: Name of the source PDF file
            
        Returns:
            List of formatted content entries with metadata
        """
        doc_title = os.path.splitext(pdf_filename)[0]
        formatted_content = []
        # State tracking variables
        current_headings = {}  # {level: heading_text}
        current_text_group = []
        current_coords = None
        current_page = 1
        current_doc_type = "text"

        # Table/Figure handling
        current_table_num = 1
        current_figure_num = 1
        current_group_type = None
        group_bounds = []

        # Skip aside groups containing prompts
        skip_aside_groups = set()

        for element in raw_json.get('elements', []):
            path = element.get('Path', '')
            text = element.get('Text', '').strip()
            page = element.get('Page', 1)  # Default to page 1 if None
            bounds = element.get('Bounds', None)
            doc_type = "text"

            # Handle aside elements (skip prompt content and mark groups)
            if 'aside' in path.lower():
                group_id = self._extract_aside_group_id(path)
                if group_id in skip_aside_groups:
                    continue
                if text.startswith("PROMPT Doc No"):
                    skip_aside_groups.add(group_id)
                    continue
            
            # Skip ANY other element that starts with PROMPT Doc (universal filter)
            elif text.startswith("PROMPT Doc No"):
                    continue

            # Determine element type
            if self.table_path_re.search(path):
                # Skip other common footer content that appears on every page
                if 'SNH' in text or 'Date loaded on PROMPT:' in text:
                    continue
                doc_type = "table"
            elif self.figure_pattern.search(path):
                doc_type = "figure"

            # Process tables and figures
            if doc_type in ["table", "figure"]:
                formatted_content, current_table_num, current_figure_num, current_group_type, group_bounds, current_page = self._handle_table_figure_element(
                    doc_type, page, bounds, current_group_type, group_bounds, current_page,
                    current_table_num, current_figure_num, formatted_content, doc_title, current_headings
                )
                continue

            # Finalize table/figure group when encountering text
            if current_group_type:
                formatted_content, current_table_num, current_figure_num = self._finalize_table_figure_group(
                    current_group_type, current_table_num, current_figure_num,
                    formatted_content, doc_title, current_headings, current_page, group_bounds
                )
                current_group_type = None
                group_bounds = []

            # Process headings
            heading_match = self.heading_pattern.search(path.lower())
            if heading_match:
                # Finalize current text group
                if current_text_group:
                    formatted_content.append(self._create_text_entry(
                        current_text_group, doc_title, current_headings, current_page, current_doc_type, current_coords
                    ))
                    current_text_group = []
                    current_coords = None

                # Update heading hierarchy
                heading_level = int(heading_match.group(1))
                current_headings = {k: v for k, v in current_headings.items() if k < heading_level}
                current_headings[heading_level] = text
                current_doc_type = "text"
                continue

            # Process text elements
            elif ('p' in path.lower() or 'l' in path.lower()) and text:
                # Skip PROMPT footer text that appears on every page
                if 'PROMPT Doc No:' in text or 'SNH' in text:
                    continue
                # Handle page changes
                if current_text_group and page != current_page:
                    formatted_content.append(self._create_text_entry(
                        current_text_group, doc_title, current_headings, current_page, current_doc_type, current_coords
                    ))
                    current_text_group = []
                    current_coords = None

                # Initialize or update text group
                if not current_text_group:
                    current_page = page
                    current_coords = bounds.copy() if bounds else None
                else:
                    if bounds and current_coords:
                        current_coords = self._merge_coordinates(current_coords, bounds)
                
                current_text_group.append(text)
                current_doc_type = "text"

        # Finalize remaining content
        if current_text_group:
            formatted_content.append(self._create_text_entry(
                current_text_group, doc_title, current_headings, current_page, current_doc_type, current_coords
            ))

        if current_group_type:
            formatted_content, _, _ = self._finalize_table_figure_group(
                current_group_type, current_table_num, current_figure_num,
                formatted_content, doc_title, current_headings, current_page, group_bounds
            )

        return formatted_content

    def _extract_aside_group_id(self, path: str) -> str:
        """Extract aside group ID from path."""
        # Match /aside[number]/ anywhere in path, not just at end
        match = re.search(r'/aside\[(\d+)\]/', path.lower())
        if match:
            return match.group(1)
        # Handle /aside/ without number (defaults to group 1)
        elif '/aside/' in path.lower():
            return "1"
        else:
            return "1"

    def _handle_table_figure_element(self, doc_type: str, page: int, bounds: List, 
                                   current_group_type: str, group_bounds: List, current_page: int,
                                   current_table_num: int, current_figure_num: int,
                                   formatted_content: List, doc_title: str, current_headings: Dict) -> Tuple:
        """Handle table and figure element processing."""
        if current_group_type == doc_type:
            # Handle page changes within same element type
            if page != current_page:
                # Finalize current group
                formatted_content, current_table_num, current_figure_num = self._finalize_table_figure_group(
                    current_group_type, current_table_num, current_figure_num,
                    formatted_content, doc_title, current_headings, current_page, group_bounds
                )
                group_bounds = bounds.copy() if bounds else []
                current_page = page
            else:
                # Merge coordinates on same page
                if bounds and group_bounds:
                    group_bounds = self._merge_coordinates(group_bounds, bounds)
        else:
            # Finalize previous group if exists
            if current_group_type:
                formatted_content, current_table_num, current_figure_num = self._finalize_table_figure_group(
                    current_group_type, current_table_num, current_figure_num,
                    formatted_content, doc_title, current_headings, current_page, group_bounds
                )
            
            # Start new group
            current_group_type = doc_type
            group_bounds = bounds.copy() if bounds else []
            current_page = page

        return formatted_content, current_table_num, current_figure_num, current_group_type, group_bounds, current_page

    def _finalize_table_figure_group(self, group_type: str, table_num: int, figure_num: int,
                                   formatted_content: List, doc_title: str, current_headings: Dict,
                                   page: int, bounds: List) -> Tuple:
        """Finalize a table or figure group."""
        content_label = f"Table {table_num}" if group_type == "table" else f"Figure {figure_num}"
        
        metadata_prefix = self.metadata_builder.create_metadata_prefix(
            doc_title,
            current_headings.get(1),
            current_headings.get(2),
            page,
            group_type,
            {k: v for k, v in current_headings.items() if k > 2}
        )

        formatted_content.append({
            "content": metadata_prefix + content_label,
            "meta": {
                "title": doc_title,
                "coordinate": bounds,
                "Doc_type": group_type,
                "headings": current_headings.copy(),
                "page": page
            }
        })

        if group_type == "table":
            table_num += 1
        else:
            figure_num += 1

        return formatted_content, table_num, figure_num

    def _create_text_entry(self, text_group: List[str], doc_title: str, current_headings: Dict,
                         page: int, doc_type: str, coords: List) -> Dict:
        """Create a text entry with metadata."""
        metadata_prefix = self.metadata_builder.create_metadata_prefix(
            doc_title,
            current_headings.get(1),
            current_headings.get(2),
            page,
            doc_type,
            {k: v for k, v in current_headings.items() if k > 2}
        )

        return {
            "content": metadata_prefix + " ".join(text_group),
            "meta": {
                "title": doc_title,
                "coordinate": coords,
                "Doc_type": doc_type,
                "headings": current_headings.copy(),
                "page": page
            }
        }

    def _merge_coordinates(self, coords1: List, coords2: List) -> List:
        """Merge two coordinate bounding boxes."""
        return [
            min(coords1[0], coords2[0]),  # left
            min(coords1[1], coords2[1]),  # top
            max(coords1[2], coords2[2]),  # right
            max(coords1[3], coords2[3])   # bottom
        ]


class OCRProcessor:
    """Handles image capture and OCR processing for tables and figures using Vertex AI."""
    
    def __init__(self, pdf_path: str, zoom: int = 3):
        self.pdf_path = pdf_path
        self.zoom = zoom
        # Use Vertex AI with Gemini 2.5 Flash
        self.model = GenerativeModel('gemini-2.5-flash')
    
    def capture_screenshot(self, page_num: int, coordinates: List[float]) -> Optional[Image.Image]:
        """
        Capture a screenshot of a PDF region.
        
        Args:
            page_num: Page number (0-indexed for PyMuPDF)
            coordinates: [x1, y1, x2, y2] coordinates
            
        Returns:
            PIL Image of the region, or None if capture fails
        """
        try:
            # Validate inputs
            if not coordinates or len(coordinates) != 4:
                raise ValueError(f"Invalid coordinates: {coordinates}")
            
            # Open PDF and get page
            doc = fitz.open(self.pdf_path)
            page = doc.load_page(page_num)
            
            # Convert coordinates
            x1, y1, x2, y2 = map(float, coordinates)
            pdf_w, pdf_h = page.rect.width, page.rect.height
            
            # Render page with zoom
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes()))
            img_w, img_h = img.size
            
            # Convert coordinates from PDF space to image pixels
            x1_px = int((x1 / pdf_w) * img_w)
            x2_px = int((x2 / pdf_w) * img_w)
            y1_px = int(img_h - ((y1 / pdf_h) * img_h))  # Flip Y axis
            y2_px = int(img_h - ((y2 / pdf_h) * img_h))
            
            # Ensure proper ordering
            left = min(x1_px, x2_px)
            right = max(x1_px, x2_px)
            top = min(y1_px, y2_px)
            bottom = max(y1_px, y2_px)
            
            # Crop and return
            cropped_img = img.crop((left, top, right, bottom))
            doc.close()
            return cropped_img
            
        except Exception as e:
            logging.error(f"Screenshot capture failed: {e}")
            return None
    
    def process_with_ocr(self, image: Image.Image, doc_type: str, 
                        document_name: str = None, headings_context: dict = None) -> Tuple[str, float]:
        """
        Process image with OCR using Gemini AI.
        
        Args:
            image: PIL Image to process
            doc_type: Type of content ('table' or 'figure')
            
        Returns:
            Tuple of (OCR text summary, confidence score)
        """
        try:
            # Save debug image in organized structure for Phase 2 - AUTO-DETECT FOLDER
            pdf_base_name = os.path.splitext(os.path.basename(self.pdf_path))[0] if hasattr(self, 'pdf_path') else "unknown"
            
            # Auto-detect folder number from PDF path (e.g., "split_folders 2/2/" → folder "2")
            folder_number = "1"  # Default fallback
            if hasattr(self, 'pdf_path') and self.pdf_path:
                path_parts = self.pdf_path.replace('\\', '/').split('/')
                for i, part in enumerate(path_parts):
                    if 'split_folders' in part and i + 1 < len(path_parts):
                        folder_number = path_parts[i + 1]
                        break
            
            debug_dir = f"ocr_input_images_split_folder2/{folder_number}/{pdf_base_name}"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_path = os.path.join(debug_dir, f"{doc_type}_{timestamp}.png")
            image.save(image_path)
            logging.info(f"Saved OCR input image: {image_path}")

            # Enhanced prompt with document context (metadata confirmed working, no need for acknowledgment)
            context_info = ""
            if document_name or headings_context:
                context_info = f"""Document context: {document_name or 'Unknown document'}"""
                if headings_context and headings_context.get('current_heading'):
                    context_info += f" | Section: {headings_context['current_heading']}"
                if headings_context and headings_context.get('page_number'):
                    context_info += f" | Page {headings_context['page_number']}"
                context_info += f" | Content type: {doc_type}\n\n"
            
            prompt = f"""{context_info}Analyze this {doc_type} and provide a detailed summary. Include relevant context, key data points and conclusions. Do not assume or include any information not found in the image.

After your analysis, provide a confidence score from 0.0 to 1.0 indicating how clear and readable the {doc_type} content is, where:
- 0.0-0.3: Very poor quality, barely readable
- 0.4-0.6: Poor quality, some content unclear
- 0.7-0.8: Good quality, mostly clear
- 0.9-1.0: Excellent quality, very clear

Format your response as:
SUMMARY: [your detailed analysis]

CONFIDENCE: [score between 0.0 and 1.0]"""

            # Convert PIL Image to bytes for Vertex AI
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()

            # Process with Vertex AI Gemini 2.5 Flash
            response = self.model.generate_content([prompt, Part.from_data(image_bytes, mime_type="image/png")])
            time.sleep(1)  # To avoid rate-limiting issues
            
            # Parse response to extract summary and confidence
            response_text = response.text
            summary = response_text
            confidence = 0.8  # Default confidence
            
            # Try to extract confidence score
            if "CONFIDENCE:" in response_text:
                parts = response_text.split("CONFIDENCE:")
                if len(parts) == 2:
                    summary = parts[0].replace("SUMMARY:", "").strip()
                    try:
                        confidence_text = parts[1].strip()
                        confidence = float(confidence_text)
                        # Ensure confidence is in valid range
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        logging.warning(f"Could not parse confidence score from: {confidence_text}")
                        confidence = 0.8
            
            return summary, confidence
            
        except Exception as e:
            logging.error(f"OCR processing failed: {e}")
            
            # Re-raise quota errors to stop the batch
            error_msg = str(e).lower()
            quota_indicators = [
                'quota', 'rate limit', 'too many requests', 'quota exceeded',
                '429', 'you exceeded your current quota'
            ]
            
            if any(indicator in error_msg for indicator in quota_indicators):
                logging.error(f"🛑 QUOTA ERROR in OCR - re-raising to stop batch: {e}")
                raise Exception(f"QUOTA_ERROR_DETECTED: {e}")
            
            return f"[OCR processing error: {str(e)}]", 0.0
    
    def process_single_element(self, entry: Dict) -> None:
        """Process a single table or figure entry with OCR."""
        try:
            meta = entry.get('meta', {})
            doc_type = meta.get('Doc_type')
            page_num = meta.get('page', 1)
            coordinates = meta.get('coordinate')
            
            if not coordinates:
                entry['content'] += "\n\n[No coordinates available for OCR]"
                # Add default confidence score
                meta['ocr_confidence'] = 0.0
                return
            
            # Try capture with original coordinates
            img = self.capture_screenshot(page_num, coordinates)
            
            # Retry with expanded coordinates if failed
            if img is None:
                logging.info("Retrying with expanded coordinates")
                expanded_coords = [
                    max(0, coordinates[0] - 10),
                    max(0, coordinates[1] - 10),
                    min(1000, coordinates[2] + 10),
                    min(1000, coordinates[3] + 10)
                ]
                img = self.capture_screenshot(page_num, expanded_coords)
            
            if img:
                img = img.convert('L')  # Convert to grayscale
                # Prepare metadata for OCR
                doc_name = os.path.basename(self.pdf_path) if hasattr(self, 'pdf_path') and self.pdf_path else "Unknown Document"
                headings = meta.get('headings', {})
                headings_info = {
                    'current_heading': headings.get(1, 'None'),  # H1 heading
                    'current_subheading': headings.get(2, 'None'),  # H2 heading  
                    'page_number': meta.get('page', entry.get('page', 'Unknown'))
                }
                summary, confidence = self.process_with_ocr(img, doc_type, doc_name, headings_info)
                entry['content'] = f"{entry['content']}\n\nSUMMARY: {summary}"
                # Store confidence score in metadata
                meta['ocr_confidence'] = confidence
                logging.info(f"OCR processed {doc_type} with confidence: {confidence:.3f}")
            else:
                entry['content'] = f"{entry['content']}\n\n[Failed to capture content]"
                meta['ocr_confidence'] = 0.0
                
        except Exception as e:
            logging.error(f"Error processing {entry.get('meta', {}).get('Doc_type', 'element')}: {str(e)}")
            entry['content'] = f"{entry['content']}\n\n[Processing error: {str(e)}]"
            meta['ocr_confidence'] = 0.0
    
    def stitch_images_vertically(self, images: List[Image.Image]) -> Optional[Image.Image]:
        """Combine multiple images vertically."""
        if not images:
            return None
        
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)
        
        combined_img = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height
        
        return combined_img
    
    def process_merged_table_group(self, group: List[Dict]) -> Dict:
        """Process a group of consecutive tables into a merged entry."""
        page_coords = []
        for entry in group:
            meta = entry.get('meta', {})
            page = meta.get('page', 1)
            coords = meta.get('coordinate')
            if coords and len(coords) == 4:
                page_coords.append((page, coords))
        
        # Capture screenshots for each part
        images = []
        for page, coords in page_coords:
            try:
                img = self.capture_screenshot(page, coords)
                if img is None:
                    # Retry with expanded coordinates
                    expanded_coords = [
                        max(0, coords[0] - 10),
                        max(0, coords[1] - 10),
                        min(1000, coords[2] + 10),
                        min(1000, coords[3] + 10)
                    ]
                    img = self.capture_screenshot(page, expanded_coords)
                if img:
                    images.append(img)
            except Exception as e:
                logging.error(f"Error capturing table part on page {page}: {e}")
        
        # Process combined image
        summary = "[No table parts captured]"
        confidence = 0.0
        if images:
            combined_img = self.stitch_images_vertically(images)
            if combined_img:
                # Prepare metadata for table OCR
                doc_name = os.path.basename(self.pdf_path) if hasattr(self, 'pdf_path') and self.pdf_path else "Unknown Document"
                first_entry_meta = group[0]['meta'] if group else {}
                headings = first_entry_meta.get('headings', {})
                headings_info = {
                    'current_heading': headings.get(1, 'None'),  # H1 heading
                    'current_subheading': headings.get(2, 'None'),  # H2 heading
                    'page_number': first_entry_meta.get('page', group[0].get('page', 'Unknown') if group else 'Unknown')
                }
                summary, confidence = self.process_with_ocr(combined_img, 'table', doc_name, headings_info)
        
        # Create merged entry
        first_meta = group[0]['meta'].copy()
        original_content = group[0]['content'].split('\n\nSUMMARY:')[0]
        merged_content = f"{original_content}\n\nSUMMARY: {summary}"
        
        # Add confidence score to metadata
        first_meta['ocr_confidence'] = confidence
        
        return {
            'content': merged_content,
            'meta': first_meta
        }


class PDFProcessor:
    """Main orchestrator class for the PDF processing pipeline."""
    
    def __init__(self, pdf_path: str, output_dir: str = "processed_output"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.extractor = AdobePDFExtractor()
        self.formatter = JSONFormatter()
        self.ocr_processor = OCRProcessor(pdf_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    # --- RAW JSON CACHE UTILS ---
    RAW_JSON_CACHE_DIR = "raw_json_cache_10pdfs/raw_extractions"
    def get_raw_json_cache_path(self, pdf_path):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        # Remove spaces and special chars for cache file
        safe_base = re.sub(r'[^A-Za-z0-9]', '', base)
        return os.path.join(self.RAW_JSON_CACHE_DIR, f"{safe_base}.json")

    def load_cached_raw_json(self, pdf_path):
        path = self.get_raw_json_cache_path(pdf_path)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_raw_json_to_cache(self, pdf_path, raw_json):
        os.makedirs(self.RAW_JSON_CACHE_DIR, exist_ok=True)
        path = self.get_raw_json_cache_path(pdf_path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False)

    def process(self) -> str:
        """
        Run the complete PDF processing pipeline.
        
        Returns:
            Path to the output JSON file
        """
        logging.info(f"Starting PDF processing: {self.pdf_path}")
        
        # Step 1: Try to load cached raw JSON
        raw_json = self.load_cached_raw_json(self.pdf_path)
        if raw_json is not None:
            logging.info("Using cached raw JSON extraction.")
        else:
            logging.info("Extracting raw JSON from Adobe API...")
            raw_json = self.extractor.extract_pdf_to_json(self.pdf_path)
            self.save_raw_json_to_cache(self.pdf_path, raw_json)
        
        # Step 2: Format JSON into structured content
        pdf_filename = os.path.basename(self.pdf_path)
        formatted_json = self.formatter.format_json(raw_json, pdf_filename)
        
        # Step 3: Process tables and figures with OCR
        logging.info("Step 3: Processing tables and figures with OCR...")
        self._process_tables_and_figures(formatted_json)
        
        # Step 4: Save final output
        logging.info("Step 4: Saving processed output...")
        output_path = self._save_output(formatted_json, pdf_filename)
        
        logging.info(f"PDF processing complete: {output_path}")
        return output_path
    
    def extract_raw_json_only(self) -> Dict:
        """Extract only the raw JSON without formatting or OCR processing."""
        return self.extractor.extract_pdf_to_json(self.pdf_path)
    
    def format_json_only(self, raw_json: Dict) -> List[Dict]:
        """Format raw JSON without OCR processing."""
        pdf_filename = os.path.basename(self.pdf_path)
        return self.formatter.format_json(raw_json, pdf_filename)
    
    def _process_tables_and_figures(self, formatted_json: List[Dict]) -> None:
        """Process tables and figures with OCR, merging consecutive tables."""
        # STEP 1: First merge text entries under same subheading across pages
        self._merge_text_across_pages(formatted_json)
        
        # STEP 2: Then process tables and figures with OCR (existing logic)
        i = 0
        while i < len(formatted_json):
            entry = formatted_json[i]
            meta = entry.get('meta', {})
            doc_type = meta.get('Doc_type')

            if doc_type == 'table':
                # Check for consecutive table merging
                current_page = meta.get('page', 1)
                group = [entry]
                j = i + 1
                
                # Collect consecutive tables on subsequent pages
                while j < len(formatted_json):
                    next_entry = formatted_json[j]
                    next_meta = next_entry.get('meta', {})
                    if (next_meta.get('Doc_type') == 'table' and 
                        next_meta.get('page', 1) == current_page + 1):
                        group.append(next_entry)
                        current_page = next_meta.get('page', 1)
                        j += 1
                    else:
                        break
                
                if len(group) > 1:
                    # Process merged table group
                    merged_entry = self.ocr_processor.process_merged_table_group(group)
                    formatted_json[i:j] = [merged_entry]
                    i += 1
                else:
                    # Process single table
                    self.ocr_processor.process_single_element(entry)
                    i += 1

            elif doc_type == 'figure':
                # Process figures individually
                self.ocr_processor.process_single_element(entry)
                i += 1
            else:
                i += 1
    
    def _merge_text_across_pages(self, formatted_json: List[Dict]) -> None:
        """Merge text entries under the same subheading when they span across pages."""
        i = 0
        while i < len(formatted_json):
            entry = formatted_json[i]
            meta = entry.get('meta', {})
            doc_type = meta.get('Doc_type')

            if doc_type == 'text':
                # Check for consecutive text under same subheading
                current_headings = meta.get('headings', {})
                current_page = meta.get('page', 1)
                group = [entry]
                j = i + 1
                
                # Collect consecutive text entries with same headings on subsequent pages
                while j < len(formatted_json):
                    next_entry = formatted_json[j]
                    next_meta = next_entry.get('meta', {})
                    next_headings = next_meta.get('headings', {})
                    
                    if (next_meta.get('Doc_type') == 'text' and 
                        next_meta.get('page', 1) == current_page + 1 and
                        self._same_heading_structure(current_headings, next_headings)):
                        group.append(next_entry)
                        current_page = next_meta.get('page', 1)
                        j += 1
                    else:
                        break
                
                if len(group) > 1:
                    # Merge text group
                    merged_entry = self._process_merged_text_group(group)
                    formatted_json[i:j] = [merged_entry]
                    i += 1
                else:
                    i += 1
            else:
                i += 1
    
    def _same_heading_structure(self, headings1: Dict, headings2: Dict) -> bool:
        """Check if two heading structures are the same."""
        # Compare all heading levels (H1, H2, H3, etc.)
        return headings1 == headings2
    
    def _process_merged_text_group(self, group: List[Dict]) -> Dict:
        """Merge a group of consecutive text entries into a single entry."""
        # Extract content from each entry (remove metadata prefix)
        content_parts = []
        all_pages = []
        all_coordinates = []
        
        for entry in group:
            content = entry['content']
            meta = entry.get('meta', {})
            
            # Extract just the content after the metadata prefix
            if '||' in content:
                content_part = content.split('||', 1)[1].strip()
            else:
                content_part = content
            
            content_parts.append(content_part)
            all_pages.append(meta.get('page'))
            if meta.get('coordinate'):
                all_coordinates.append(meta.get('coordinate'))
        
        # Create merged content
        first_meta = group[0]['meta'].copy()
        merged_content_text = ' '.join(content_parts)
        
        # Create page range
        pages = [p for p in all_pages if p is not None]
        if len(pages) > 1:
            page_info = f"{min(pages)}-{max(pages)}"
        else:
            page_info = pages[0] if pages else 1
        
        # Recreate metadata prefix with page range
        metadata_prefix = self.formatter.metadata_builder.create_metadata_prefix(
            first_meta['title'],
            first_meta['headings'].get(1),
            first_meta['headings'].get(2),
            page_info,  # Use page range instead of single page
            first_meta['Doc_type'],
            {k: v for k, v in first_meta['headings'].items() if k > 2}
        )
        
        return {
            'content': metadata_prefix + merged_content_text,
            'meta': {
                **first_meta,
                'page': page_info,
                'coordinate': all_coordinates,
                'is_merged_text': True,
                'original_page_count': len(pages)
            }
        }
    
    def _save_output(self, formatted_json: List[Dict], pdf_filename: str) -> str:
        """Save the processed JSON to file."""
        pdf_name = os.path.splitext(pdf_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pdf_name}_processed_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, "w", encoding='utf-8') as output_file:
            json.dump(formatted_json, output_file, indent=4, ensure_ascii=False)
        
        logging.info(f"Processed JSON saved to: {output_path}")
        return output_path


# Convenience functions for backward compatibility
def extract_pdf_to_json(pdf_path: str) -> Dict:
    """Extract raw JSON from PDF using Adobe services."""
    extractor = AdobePDFExtractor()
    return extractor.extract_pdf_to_json(pdf_path)

def format_json(raw_json: Dict, pdf_filename: str) -> List[Dict]:
    """Format raw Adobe JSON into structured content."""
    formatter = JSONFormatter()
    return formatter.format_json(raw_json, pdf_filename)

def process_pdf_complete(pdf_path: str, output_dir: str = "processed_output") -> str:
    """Process a PDF through the complete pipeline."""
    processor = PDFProcessor(pdf_path, output_dir)
    return processor.process()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_processor_organized.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    try:
        output_path = process_pdf_complete(pdf_path)
        print(f"Processing complete! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1) 