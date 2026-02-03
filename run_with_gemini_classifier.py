#!/usr/bin/env python3
"""
Process PDF with Gemini-based figure classification.
Uses Gemini API to classify images as INFORMATIVE or DECORATIVE before OCR.

Usage:
    python run_with_gemini_classifier.py path/to/your.pdf [output_dir]
    
Environment Variables:
    GOOGLE_API_KEY: Your Google Generative AI API key (free tier)
    USE_FREE_GEMINI: Set to "true" to use free Gemini API (default: true)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / 'config' / '.env')

from pdf_processor_organized import (
    AdobePDFExtractor, 
    JSONFormatter,
)
import fitz  # PyMuPDF
from PIL import Image
import io

# Gemini API for classification
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False


# ============================================
# GEMINI CLASSIFIER
# ============================================

def classify_with_gemini(image: Image.Image, doc_type: str = 'figure') -> tuple[bool, str, float]:
    """
    Use Gemini to classify an image as INFORMATIVE or DECORATIVE.
    
    Args:
        image: PIL Image to classify
        doc_type: Type of content ('table' or 'figure')
    
    Returns:
        Tuple of (is_informative: bool, reason: str, confidence: float)
    """
    if not GEMINI_AVAILABLE:
        return True, "gemini_not_available", 0.5
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not set. Falling back to size-based filtering.")
        return True, "no_api_key", 0.5
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Fast and cheap
        
        # Simple classification prompt
        prompt = f"""Look at this {doc_type} from a medical/clinical guideline PDF.

Classify it as either:
- INFORMATIVE: Contains important clinical information, data, flowcharts, diagrams with text, tables, decision trees, or medical illustrations
- DECORATIVE: Just decorative images, photos, logos, icons, or visual elements without meaningful content

Respond in this exact format:
CLASSIFICATION: [INFORMATIVE or DECORATIVE]
REASON: [one sentence explaining why]
CONFIDENCE: [0.0 to 1.0]"""

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # Call Gemini
        response = model.generate_content([prompt, Image.open(img_bytes)])
        response_text = response.text.strip()
        
        # Parse response
        is_informative = False
        reason = "unknown"
        confidence = 0.5
        
        if "CLASSIFICATION:" in response_text:
            classification_line = [l for l in response_text.split('\n') if 'CLASSIFICATION:' in l][0]
            if 'INFORMATIVE' in classification_line.upper():
                is_informative = True
        
        if "REASON:" in response_text:
            reason_line = [l for l in response_text.split('\n') if 'REASON:' in l]
            if reason_line:
                reason = reason_line[0].split('REASON:')[1].strip()
        
        if "CONFIDENCE:" in response_text:
            conf_line = [l for l in response_text.split('\n') if 'CONFIDENCE:' in l]
            if conf_line:
                try:
                    conf_str = conf_line[0].split('CONFIDENCE:')[1].strip()
                    confidence = float(conf_str)
                except:
                    pass
        
        return is_informative, reason, confidence
        
    except Exception as e:
        print(f"⚠️  Gemini classification error: {e}")
        # Fallback: assume informative if error
        return True, f"classification_error: {str(e)[:50]}", 0.5


def calculate_figure_metrics(coordinates, page_width=612, page_height=792):
    """Calculate metrics for fallback filtering."""
    if not coordinates or len(coordinates) != 4:
        return None
    
    x1, y1, x2, y2 = map(float, coordinates)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    
    return {
        'width': width,
        'height': height,
        'area': area,
    }


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


def process_with_gemini_classifier(pdf_path: str, output_dir: str = "output"):
    """Process PDF with Gemini-based figure classification."""
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = pdf_path.stem
    
    # Create screenshot directories
    important_dir = Path("screenshots") / pdf_name / "important"
    skipped_dir = Path("screenshots") / pdf_name / "skipped"
    important_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🤖 Processing PDF with GEMINI CLASSIFIER")
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📁 Output: {output_dir}")
    print(f"📸 Important: {important_dir}")
    print(f"🚫 Skipped: {skipped_dir}")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GOOGLE_API_KEY not set in .env")
        print("   Add: GOOGLE_API_KEY=your_key_here")
        print("   Falling back to size-based filtering...\n")
    else:
        print(f"\n✅ Using Gemini API for classification")
    
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
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False)
        print(f"   Cached to: {cache_file}")
    
    # Step 2: Format JSON
    print("\n📝 Formatting JSON...")
    formatter = JSONFormatter()
    formatted_json = formatter.format_json(raw_json, pdf_path.name)
    print(f"   Found {len(formatted_json)} content entries")
    
    # Get page dimensions from PDF
    doc = fitz.open(str(pdf_path))
    page_dims = {}
    for i in range(len(doc)):
        page = doc.load_page(i)
        page_dims[i] = (page.rect.width, page.rect.height)
    doc.close()
    
    # Step 3: Classify figures with Gemini
    print("\n🤖 Classifying figures with Gemini...")
    
    stats = {
        'tables_important': 0,
        'tables_skipped': 0,
        'figures_important': 0,
        'figures_skipped': 0,
        'classification_errors': 0,
    }
    classification_details = []
    
    figures_to_classify = [
        (entry, meta, doc_type, page_num, coordinates)
        for entry in formatted_json
        for meta in [entry.get('meta', {})]
        for doc_type in [meta.get('Doc_type', '')]
        for page_num in [meta.get('page', 0)]
        for coordinates in [meta.get('coordinate')]
        if doc_type in ['table', 'figure'] and coordinates
    ]
    
    total_figures = len(figures_to_classify)
    print(f"   Found {total_figures} figures/tables to classify")
    
    for idx, (entry, meta, doc_type, page_num, coordinates) in enumerate(figures_to_classify, 1):
        print(f"\n   [{idx}/{total_figures}] Processing {doc_type} on page {page_num + 1}...", end=" ", flush=True)
        
        # Capture screenshot
        img = capture_screenshot(str(pdf_path), page_num, coordinates)
        if not img:
            print("❌ Screenshot failed")
            continue
        
        # Classify with Gemini
        is_informative, reason, confidence = classify_with_gemini(img, doc_type)
        
        # Update stats
        if doc_type == 'table':
            if is_informative:
                stats['tables_important'] += 1
            else:
                stats['tables_skipped'] += 1
        else:
            if is_informative:
                stats['figures_important'] += 1
            else:
                stats['figures_skipped'] += 1
        
        # Save classification details
        classification_details.append({
            'type': doc_type,
            'page': page_num + 1,
            'is_informative': is_informative,
            'reason': reason,
            'confidence': confidence,
        })
        
        # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        type_label = 'table' if doc_type == 'table' else 'figure'
        
        if is_informative:
            count = stats['tables_important'] if doc_type == 'table' else stats['figures_important']
            img_filename = f"{type_label}_{count}_page{page_num + 1}_{timestamp}.png"
            img_path = important_dir / img_filename
            
            entry['content'] = f"{entry['content']}\n\n[SCREENSHOT SAVED: {img_filename} - OCR pending]"
            meta['screenshot_path'] = str(img_path)
            meta['ocr_status'] = 'pending'
            meta['classification'] = 'informative'
            meta['classification_method'] = 'gemini'
            meta['classification_reason'] = reason
            meta['classification_confidence'] = confidence
            
            print(f"✅ INFORMATIVE (conf: {confidence:.2f})")
        else:
            count = stats['tables_skipped'] if doc_type == 'table' else stats['figures_skipped']
            img_filename = f"SKIP_{type_label}_{count}_page{page_num + 1}_{reason[:30]}.png"
            img_path = skipped_dir / img_filename
            
            entry['content'] = f"{entry['content']}\n\n[DECORATIVE - SKIPPED: {reason}]"
            meta['screenshot_path'] = str(img_path)
            meta['ocr_status'] = 'skipped'
            meta['classification'] = 'decorative'
            meta['classification_method'] = 'gemini'
            meta['classification_reason'] = reason
            meta['classification_confidence'] = confidence
            
            print(f"🚫 DECORATIVE (conf: {confidence:.2f})")
        
        img.save(img_path)
        
        # Rate limiting (Gemini free tier: ~15 requests/minute)
        time.sleep(4)  # Conservative delay
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\n📊 TABLES:")
    print(f"   ✅ Informative (will OCR): {stats['tables_important']}")
    print(f"   🚫 Decorative (skipped): {stats['tables_skipped']}")
    
    print(f"\n🖼️ FIGURES:")
    print(f"   ✅ Informative (will OCR): {stats['figures_important']}")
    print(f"   🚫 Decorative (skipped): {stats['figures_skipped']}")
    
    # Confidence statistics
    if classification_details:
        avg_confidence = sum(d['confidence'] for d in classification_details) / len(classification_details)
        print(f"\n📈 Classification Statistics:")
        print(f"   Average confidence: {avg_confidence:.2f}")
        informative_conf = [d['confidence'] for d in classification_details if d['is_informative']]
        decorative_conf = [d['confidence'] for d in classification_details if not d['is_informative']]
        if informative_conf:
            print(f"   Informative avg confidence: {sum(informative_conf)/len(informative_conf):.2f}")
        if decorative_conf:
            print(f"   Decorative avg confidence: {sum(decorative_conf)/len(decorative_conf):.2f}")
    
    total_important = stats['tables_important'] + stats['figures_important']
    total_skipped = stats['tables_skipped'] + stats['figures_skipped']
    total = total_important + total_skipped
    
    print(f"\n💰 OCR COST SAVINGS:")
    print(f"   Total figures: {total}")
    if total > 0:
        print(f"   Will OCR: {total_important} ({100*total_important/total:.1f}%)")
        print(f"   Skipped: {total_skipped} ({100*total_skipped/total:.1f}%)")
        print(f"   Estimated savings: ~${total_skipped * 0.01:.2f} (at $0.01/image)")
    
    # Step 4: Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{pdf_name}_processed_GEMINI_CLASSIFIED_{timestamp}.json"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_json, f, indent=4, ensure_ascii=False)
    
    # Save classification report
    report_path = output_dir / f"{pdf_name}_classification_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pdf_name': pdf_name,
            'stats': stats,
            'classifications': classification_details,
            'timestamp': timestamp,
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   📄 Output: {output_path}")
    print(f"   📊 Classification report: {report_path}")
    print(f"   📸 Important figures: {important_dir}/")
    print(f"   🚫 Skipped figures: {skipped_dir}/ (for review)")
    
    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_gemini_classifier.py <pdf_path> [output_dir]")
        print("\nThis script uses Gemini API to classify figures as informative or decorative.")
        print("\nRequired environment variable:")
        print("  GOOGLE_API_KEY: Your Google Generative AI API key")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        output_path = process_with_gemini_classifier(pdf_path, output_dir)
        print(f"\n🎉 Done! Check: {output_path}")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
