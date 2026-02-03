#!/usr/bin/env python3
"""
Process PDF WITHOUT OCR - with smart filtering of decorative figures.
Only captures meaningful figures (large enough, in content area, not header/footer).

Usage:
    python run_without_ocr_filtered.py path/to/your.pdf [output_dir]
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
)
import json
import fitz  # PyMuPDF
from PIL import Image
import io


# ============================================
# FIGURE FILTERING LOGIC
# ============================================

def calculate_figure_metrics(coordinates, page_width=612, page_height=792):
    """Calculate metrics to determine if figure is important."""
    if not coordinates or len(coordinates) != 4:
        return None
    
    x1, y1, x2, y2 = map(float, coordinates)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height
    aspect_ratio = width / height if height > 0 else 0
    
    # Center position
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Relative position (0-1 scale)
    rel_center_y = center_y / page_height
    
    return {
        'width': width,
        'height': height,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'center_x': center_x,
        'center_y': center_y,
        'rel_center_y': rel_center_y,
        'in_header': rel_center_y < 0.1,
        'in_footer': rel_center_y > 0.9,
    }


def classify_figure(metrics, doc_type='figure'):
    """
    Classify figure as IMPORTANT or DECORATIVE.
    
    Returns: tuple (is_important: bool, reason: str)
    """
    if metrics is None:
        return False, "no_metrics"
    
    # Tables are almost always important
    if doc_type == 'table':
        if metrics['area'] > 5000:  # Reasonable size table
            return True, "table"
        return False, "tiny_table"
    
    # Figure classification rules
    rules_failed = []
    
    # Rule 1: Size thresholds
    MIN_WIDTH = 120
    MIN_HEIGHT = 100
    MIN_AREA = 20000  # ~200x100 pixels minimum
    
    if metrics['width'] < MIN_WIDTH:
        rules_failed.append(f"too_narrow({metrics['width']:.0f}<{MIN_WIDTH})")
    if metrics['height'] < MIN_HEIGHT:
        rules_failed.append(f"too_short({metrics['height']:.0f}<{MIN_HEIGHT})")
    if metrics['area'] < MIN_AREA:
        rules_failed.append(f"too_small({metrics['area']:.0f}<{MIN_AREA})")
    
    # Rule 2: Extreme aspect ratios (banners, thin strips)
    if metrics['aspect_ratio'] > 8 or metrics['aspect_ratio'] < 0.1:
        rules_failed.append(f"extreme_aspect({metrics['aspect_ratio']:.2f})")
    
    # Rule 3: Header/footer decorations (small elements at top/bottom)
    if metrics['in_header'] or metrics['in_footer']:
        if metrics['area'] < 50000:  # Small header/footer element
            rules_failed.append("header_footer_decoration")
    
    # Rule 4: Very small square elements (likely icons)
    if metrics['width'] < 80 and metrics['height'] < 80:
        rules_failed.append("icon_sized")
    
    if rules_failed:
        return False, "|".join(rules_failed)
    
    return True, "passed_all_rules"


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


def process_with_filtering(pdf_path: str, output_dir: str = "output"):
    """Process PDF with smart filtering of decorative figures."""
    
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
    print("🔧 Processing PDF with SMART FILTERING (No OCR)")
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📁 Output: {output_dir}")
    print(f"📸 Important: {important_dir}")
    print(f"🚫 Skipped: {skipped_dir}")
    print("=" * 70)
    
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
    
    # Step 3: Classify and capture figures
    print("\n🔍 Classifying figures...")
    
    stats = {
        'tables_important': 0,
        'tables_skipped': 0,
        'figures_important': 0,
        'figures_skipped': 0,
    }
    skipped_reasons = {}
    
    for entry in formatted_json:
        meta = entry.get('meta', {})
        doc_type = meta.get('Doc_type', '')
        page_num = meta.get('page', 0)
        coordinates = meta.get('coordinate')
        
        if doc_type not in ['table', 'figure']:
            continue
        
        # Get page dimensions
        page_w, page_h = page_dims.get(page_num, (612, 792))
        
        # Calculate metrics and classify
        metrics = calculate_figure_metrics(coordinates, page_w, page_h)
        is_important, reason = classify_figure(metrics, doc_type)
        
        # Update stats
        if doc_type == 'table':
            if is_important:
                stats['tables_important'] += 1
            else:
                stats['tables_skipped'] += 1
        else:
            if is_important:
                stats['figures_important'] += 1
            else:
                stats['figures_skipped'] += 1
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        
        # Capture screenshot
        if coordinates:
            img = capture_screenshot(str(pdf_path), page_num, coordinates)
            
            if img:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                type_label = 'table' if doc_type == 'table' else 'figure'
                
                if is_important:
                    count = stats['tables_important'] if doc_type == 'table' else stats['figures_important']
                    img_filename = f"{type_label}_{count}_page{page_num}_{timestamp}.png"
                    img_path = important_dir / img_filename
                    
                    entry['content'] = f"{entry['content']}\n\n[SCREENSHOT SAVED: {img_filename} - OCR pending]"
                    meta['screenshot_path'] = str(img_path)
                    meta['ocr_status'] = 'pending'
                    meta['classification'] = 'important'
                else:
                    # Still save to skipped folder for review
                    count = stats['tables_skipped'] if doc_type == 'table' else stats['figures_skipped']
                    img_filename = f"SKIP_{type_label}_{count}_page{page_num}_{reason[:20]}.png"
                    img_path = skipped_dir / img_filename
                    
                    entry['content'] = f"{entry['content']}\n\n[DECORATIVE - SKIPPED: {reason}]"
                    meta['screenshot_path'] = str(img_path)
                    meta['ocr_status'] = 'skipped'
                    meta['classification'] = 'decorative'
                    meta['skip_reason'] = reason
                
                img.save(img_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\n📊 TABLES:")
    print(f"   ✅ Important (will OCR): {stats['tables_important']}")
    print(f"   🚫 Skipped (decorative): {stats['tables_skipped']}")
    
    print(f"\n🖼️ FIGURES:")
    print(f"   ✅ Important (will OCR): {stats['figures_important']}")
    print(f"   🚫 Skipped (decorative): {stats['figures_skipped']}")
    
    if skipped_reasons:
        print(f"\n🔍 Skip reasons breakdown:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            print(f"   - {reason}: {count}")
    
    total_important = stats['tables_important'] + stats['figures_important']
    total_skipped = stats['tables_skipped'] + stats['figures_skipped']
    total = total_important + total_skipped
    
    print(f"\n💰 OCR COST SAVINGS:")
    print(f"   Total figures: {total}")
    print(f"   Will OCR: {total_important} ({100*total_important/total:.1f}%)" if total > 0 else "   Will OCR: 0")
    print(f"   Skipped: {total_skipped} ({100*total_skipped/total:.1f}%)" if total > 0 else "   Skipped: 0")
    
    # Step 4: Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{pdf_name}_processed_FILTERED_{timestamp}.json"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_json, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   📄 Output: {output_path}")
    print(f"   📸 Important figures: {important_dir}/")
    print(f"   🚫 Skipped figures: {skipped_dir}/ (for review)")
    
    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_without_ocr_filtered.py <pdf_path> [output_dir]")
        print("\nThis script filters out decorative figures to reduce OCR costs.")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        output_path = process_with_filtering(pdf_path, output_dir)
        print(f"\n🎉 Done! Check: {output_path}")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

