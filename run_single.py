#!/usr/bin/env python3
"""
Process a single PDF file.

Usage:
    python run_single.py path/to/your.pdf [output_dir]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_processor_organized import process_pdf_complete

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single.py <pdf_path> [output_dir]")
        print("\nExample:")
        print("  python run_single.py document.pdf output/")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"📄 Processing: {pdf_path}")
    print(f"📁 Output dir: {output_dir}")
    print()
    
    try:
        output_path = process_pdf_complete(pdf_path, output_dir)
        print(f"\n✅ Success! Output saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

