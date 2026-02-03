#!/usr/bin/env python3
"""
Process PDF WITHOUT OCR - saves screenshots but skips Gemini API calls.

Usage:
    python run_without_ocr.py path/to/your.pdf [output_dir]
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'config' / '.env')

from pdf_processor_organized import (
    AdobePDFExtractor, 
    JSONFormatter,
    OCRProcessor
)
import json
import fitz  # PyMuPDF
from PIL import Image
import io


def capture_screenshot(pdf_path: str, page_num: int, coordinates: list, zoom: int = 3):
    """Capture a screenshot of a PDF region."""
    try:
        if not coordinates or len(coordinates) != 4:
            return None
        
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        x1, y1, x2, y2 = map(float, coordinates)
        pdf_w, pdf_h = page.rect.width, page.rect.height
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_w, img_h = img.size
        
        x1_px = int((x1 / pdf_w) * img_w)
        x2_px = int((x2 / pdf_w) * img_w)
        y1_px = int(img_h - ((y1 / pdf_h) * img_h))
        y2_px = int(img_h - ((y2 / pdf_h) * img_h))
        
        left = min(x1_px, x2_px)
        right = max(x1_px, x2_px)
        top = min(y1_px, y2_px)
        bottom = max(y1_px, y2_px)
        
        cropped_img = img.crop((left, top, right, bottom))
        doc.close()
        return cropped_img
        
    except Exception as e:
        print(f"⚠️ Screenshot capture failed: {e}")
        return None


def process_without_ocr(pdf_path: str, output_dir: str = "output"):
    """Process PDF without OCR - saves screenshots but skips Gemini."""
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = pdf_path.stem
    
    # Create screenshot directory
    screenshot_dir = Path("screenshots") / pdf_name
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 Processing PDF WITHOUT OCR")
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📁 Output: {output_dir}")
    print(f"📸 Screenshots: {screenshot_dir}")
    print("=" * 60)
    
    # Step 1: Check for cached raw JSON or extract
    extractor = AdobePDFExtractor()
    cache_dir = Path("raw_json_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{pdf_name}.json"
    
    if cache_file.exists():
        print("\n✅ Using cached raw JSON")
        with open(cache_file, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
    else:
        print("\n📥 Extracting with Adobe PDF Services...")
        raw_json = extractor.extract_pdf_to_json(str(pdf_path))
        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False)
        print(f"   Cached to: {cache_file}")
    
    # Step 2: Format JSON
    print("\n📝 Formatting JSON...")
    formatter = JSONFormatter()
    formatted_json = formatter.format_json(raw_json, pdf_path.name)
    print(f"   Found {len(formatted_json)} content entries")
    
    # Step 3: Capture screenshots for tables and figures (NO OCR)
    print("\n📸 Capturing screenshots (NO OCR)...")
    table_count = 0
    figure_count = 0
    
    for entry in formatted_json:
        meta = entry.get('meta', {})
        doc_type = meta.get('Doc_type', '')
        page_num = meta.get('page', 0)
        coordinates = meta.get('coordinate')
        
        if doc_type in ['table', 'figure'] and coordinates:
            img = capture_screenshot(str(pdf_path), page_num, coordinates)
            
            if img:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                if doc_type == 'table':
                    table_count += 1
                    img_filename = f"table_{table_count}_page{page_num}_{timestamp}.png"
                else:
                    figure_count += 1
                    img_filename = f"figure_{figure_count}_page{page_num}_{timestamp}.png"
                
                img_path = screenshot_dir / img_filename
                img.save(img_path)
                
                # Add placeholder content (no OCR summary)
                entry['content'] = f"{entry['content']}\n\n[SCREENSHOT SAVED: {img_filename} - OCR pending]"
                meta['screenshot_path'] = str(img_path)
                meta['ocr_status'] = 'pending'
    
    print(f"   📊 Tables captured: {table_count}")
    print(f"   🖼️  Figures captured: {figure_count}")
    
    # Step 4: Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{pdf_name}_processed_NO_OCR_{timestamp}.json"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_json, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   📄 Output: {output_path}")
    print(f"   📸 Screenshots: {screenshot_dir}/")
    print(f"\n💡 To run OCR later, process the screenshots with Gemini separately.")
    
    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_without_ocr.py <pdf_path> [output_dir]")
        print("\nExample:")
        print("  python run_without_ocr.py guidelines/my_guideline.pdf output/")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        output_path = process_without_ocr(pdf_path, output_dir)
        print(f"\n🎉 Done! Check: {output_path}")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

