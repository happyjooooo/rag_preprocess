#!/usr/bin/env python3
"""
Batch process multiple PDFs using multiprocessing.

Usage:
    python run_batch.py --pdf-dir path/to/pdfs --raw-json-dir path/to/raw_json --output-dir output/ --workers 4
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase2_multiprocessing_WORKING import Phase2MultiprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs for RAG pipeline")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--raw-json-dir", required=True, help="Directory containing raw JSON files")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.pdf_dir):
        print(f"❌ Error: PDF directory not found: {args.pdf_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.raw_json_dir):
        print(f"❌ Error: Raw JSON directory not found: {args.raw_json_dir}")
        sys.exit(1)
    
    print("🚀 Starting Batch Processing Pipeline")
    print(f"📁 PDF dir: {args.pdf_dir}")
    print(f"📁 Raw JSON dir: {args.raw_json_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"👷 Workers: {args.workers}")
    print()
    
    pipeline = Phase2MultiprocessingPipeline(
        raw_json_dir=args.raw_json_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    summary = pipeline.run_pipeline()
    
    if summary['status'] == 'completed':
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successful: {summary['successful']}/{summary['total_files']}")
        print(f"❌ Failed: {summary['failed']}/{summary['total_files']}")
    else:
        print(f"\n❌ Pipeline failed: {summary.get('message', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

