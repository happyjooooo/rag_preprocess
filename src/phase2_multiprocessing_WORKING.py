#!/usr/bin/env python3
"""
Phase 2: Enhanced Multiprocessing Pipeline - WORKING VERSION
=============================================================
Based on the SUCCESSFUL Nuclear Medicine processing pattern.

This script processes raw JSON files using the proven PDFProcessor 
with fixed metadata passing and organized OCR screenshot saving.

Features:
✅ Fixed metadata passing (headings + page numbers)
✅ Table grouping and merging across pages  
✅ OCR screenshots saved to organized directory
✅ Text merging across pages under same headings
✅ Comprehensive progress tracking
✅ Robust error handling with detailed logging
"""

import sys
import os
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import traceback
from tqdm import tqdm

# Import from same directory
from pdf_processor_organized import PDFProcessor

class Phase2MultiprocessingPipeline:
    """Enhanced multiprocessing pipeline using the proven PDFProcessor."""
    
    def __init__(self, 
                 raw_json_dir: str,
                 pdf_dir: str, 
                 output_dir: str,
                 num_workers: int = None):
        """
        Initialize the multiprocessing pipeline.
        
        Args:
            raw_json_dir: Directory containing raw JSON files
            pdf_dir: Directory containing PDF files 
            output_dir: Directory for structured JSON output
            num_workers: Number of worker processes (default: CPU count)
        """
        self.raw_json_dir = Path(raw_json_dir)
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or mp.cpu_count()
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized pipeline with {self.num_workers} workers")
        self.logger.info(f"Raw JSON dir: {self.raw_json_dir}")
        self.logger.info(f"PDF dir: {self.pdf_dir}")
        self.logger.info(f"Output dir: {self.output_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('logs/phase2_working_pipeline')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'phase2_working_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to: {log_file}")
    
    def find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching raw JSON and PDF file pairs - WITH CACHING: Skip processed files.
        
        Returns:
            List of (raw_json_path, pdf_path) tuples for unprocessed files only
        """
        pairs = []
        processed_count = 0
        
        # CORRECT: Start with PDF files (not JSON files!)
        pdf_files = list(self.pdf_dir.glob('*.pdf'))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            # Convert PDF name to expected JSON name
            # Rule: spaces → underscores, keep existing underscores
            base_name = pdf_file.stem  # Remove .pdf extension
            expected_json_name = base_name.replace(' ', '_') + '.json'
            
            # Look for the JSON file
            json_path = self.raw_json_dir / expected_json_name
            
            if not json_path.exists():
                self.logger.warning(f"No JSON found for PDF: {pdf_file.name}")
                continue
            
            # CHECK CACHING: Skip already processed files
            if self._is_file_processed(pdf_file, json_path):
                processed_count += 1
                self.logger.debug(f"CACHED (skip): {pdf_file.name}")
                continue
            
            # This file is unprocessed - add it
            pairs.append((json_path, pdf_file))
            self.logger.debug(f"UNPROCESSED: {json_path.name} ← → {pdf_file.name}")
        
        self.logger.info(f"Caching results: {processed_count} already processed, {len(pairs)} need processing")
        return pairs
    
    def _is_file_processed(self, pdf_file: Path, json_file: Path) -> bool:
        """Check if a file has already been processed by looking for output files."""
        # Check for existing processed output files
        base_name = json_file.stem  # Remove .json
        existing_outputs = list(self.output_dir.glob(f"*{base_name}*"))
        
        # Also check for OCR screenshots (auto-detect folder from PDF path)
        folder_number = "1"  # Updated for folder 1 reprocessing
        if str(pdf_file):
            path_parts = str(pdf_file).replace('\\', '/').split('/')
            for i, part in enumerate(path_parts):
                if 'split_folders' in part and i + 1 < len(path_parts):
                    folder_number = path_parts[i + 1]
                    break
        
        pdf_base_name = pdf_file.stem
        ocr_dir = Path(f'ocr_input_images_split_folder2/{folder_number}/{pdf_base_name}')
        has_screenshots = ocr_dir.exists() and list(ocr_dir.glob('*.png'))
        
        return len(existing_outputs) > 0 or bool(has_screenshots)
    
    def process_single_file(self, args: Tuple[Path, Path]) -> Dict:
        """
        Process a single raw JSON file using the proven PDFProcessor pattern.
        
        Args:
            args: Tuple of (raw_json_path, pdf_path)
            
        Returns:
            Processing result dictionary
        """
        raw_json_path, pdf_path = args
        worker_id = mp.current_process().name
        
        result = {
            'raw_json_file': str(raw_json_path),
            'pdf_file': str(pdf_path),
            'worker_id': worker_id,
            'status': 'pending',
            'timestamp_start': datetime.now().isoformat(),
            'error': None,
            'output_file': None,
            'stats': {}
        }
        
        try:
            self.logger.info(f"[{worker_id}] Processing: {raw_json_path.name}")
            
            # Load raw JSON - EXACT pattern from successful processing
            with open(raw_json_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
            
            raw_elements = len(raw_json.get('elements', []))
            self.logger.info(f"[{worker_id}] Raw elements: {raw_elements}")
            result['stats']['raw_elements'] = raw_elements
            
            # Initialize the PROVEN PDFProcessor
            processor = PDFProcessor(str(pdf_path), str(self.output_dir))
            
            # Step 1: Format JSON - EXACT pattern from successful processing
            formatted_json = processor.formatter.format_json(raw_json, pdf_path.name)
            
            # Count before processing
            text_before = sum(1 for e in formatted_json if e.get('meta', {}).get('Doc_type') == 'text')
            tables_before = sum(1 for e in formatted_json if e.get('meta', {}).get('Doc_type') == 'table')
            figures_before = sum(1 for e in formatted_json if e.get('meta', {}).get('Doc_type') == 'figure')
            
            self.logger.info(f"[{worker_id}] Before processing: {text_before} text, {tables_before} tables, {figures_before} figures")
            
            # Step 2-6: COMPLETE processing - EXACT pattern from successful processing
            # This does ALL the processing: text merging, table merging, OCR, everything
            processor._process_tables_and_figures(formatted_json)
            
            # Count final results
            text_after = sum(1 for e in formatted_json if e.get('meta', {}).get('Doc_type') == 'text')
            tables_after = sum(1 for e in formatted_json if e.get('meta', {}).get('Doc_type') == 'table')
            ocr_entries = sum(1 for e in formatted_json if 'SUMMARY:' in e.get('content', ''))
            confidence_scores = [e.get('meta', {}).get('ocr_confidence') for e in formatted_json if 'ocr_confidence' in e.get('meta', {})]
            
            # Calculate average confidence
            avg_confidence = None
            if confidence_scores and any(c for c in confidence_scores if c):
                valid_scores = [c for c in confidence_scores if c]
                avg_confidence = sum(valid_scores) / len(valid_scores)
            
            # Save final output
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{raw_json_path.stem}_PROCESSED_{timestamp}.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_json, f, indent=2, ensure_ascii=False)
            
            # Update result
            result.update({
                'status': 'success',
                'timestamp_end': datetime.now().isoformat(),
                'output_file': str(output_path),
                'stats': {
                    'raw_elements': raw_elements,
                    'text_before': text_before,
                    'tables_before': tables_before,
                    'figures_before': figures_before,
                    'text_after': text_after,
                    'tables_after': tables_after,
                    'ocr_entries': ocr_entries,
                    'avg_confidence': avg_confidence,
                    'final_entries': len(formatted_json)
                }
            })
            
            self.logger.info(f"[{worker_id}] ✅ SUCCESS: {output_filename}")
            self.logger.info(f"[{worker_id}] Final: {text_after} text, {tables_after} tables, {ocr_entries} OCR entries")
            if avg_confidence:
                self.logger.info(f"[{worker_id}] Average OCR confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            error_msg = f"Error processing {raw_json_path.name}: {str(e)}"
            self.logger.error(f"[{worker_id}] ❌ {error_msg}")
            self.logger.error(f"[{worker_id}] Traceback: {traceback.format_exc()}")
            
            result.update({
                'status': 'error',
                'timestamp_end': datetime.now().isoformat(),
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
        
        return result
    
    def run_pipeline(self) -> Dict:
        """
        Run the complete multiprocessing pipeline.
        
        Returns:
            Summary results dictionary
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting Phase 2 Working Pipeline")
        
        # Find file pairs
        file_pairs = self.find_file_pairs()
        if not file_pairs:
            self.logger.error("No file pairs found to process")
            return {'status': 'error', 'message': 'No file pairs found'}
        
        # Process files with multiprocessing
        self.logger.info(f"Processing {len(file_pairs)} files with {self.num_workers} workers")
        
        results = []
        with mp.Pool(processes=self.num_workers) as pool:
            # Use tqdm for progress tracking
            with tqdm(total=len(file_pairs), desc="Processing files") as pbar:
                for result in pool.imap(self.process_single_file, file_pairs):
                    results.append(result)
                    pbar.update(1)
                    
                    # Log progress
                    status = result['status']
                    filename = Path(result['raw_json_file']).name
                    if status == 'success':
                        self.logger.info(f"✅ Completed: {filename}")
                    else:
                        self.logger.error(f"❌ Failed: {filename}")
        
        # Generate summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = len(results) - success_count
        
        summary = {
            'status': 'completed',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_files': len(file_pairs),
            'successful': success_count,
            'failed': error_count,
            'success_rate': success_count / len(file_pairs) if file_pairs else 0,
            'results': results
        }
        
        # Save summary report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.output_dir / f'phase2_working_summary_{timestamp}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🎉 Pipeline completed!")
        self.logger.info(f"📊 Results: {success_count}/{len(file_pairs)} successful ({summary['success_rate']:.1%})")
        self.logger.info(f"⏱️ Duration: {duration}")
        self.logger.info(f"📄 Summary saved: {summary_file}")
        
        return summary

def main():
    """Main execution function."""
    print("🚀 Phase 2: Enhanced Multiprocessing Pipeline - WORKING VERSION")
    print("=" * 65)
    
    # Configuration - FOLDER 5 PROCESSING
    config = {
        'raw_json_dir': 'phase1_raw_json_output_split_folder2/10/raw_extractions',
        'pdf_dir': 'split_folders 2/10',
        'output_dir': 'processed_output_split_folder2/10',
        'num_workers': 4  # Adjust based on your system
    }
    
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Verify directories exist
    for key, path in config.items():
        if key.endswith('_dir') and not os.path.exists(path):
            print(f"❌ Error: Directory does not exist: {path}")
            return
    
    # Initialize and run pipeline
    pipeline = Phase2MultiprocessingPipeline(**config)
    summary = pipeline.run_pipeline()
    
    if summary['status'] == 'completed':
        print(f"\n🎉 Processing completed successfully!")
        print(f"✅ Successful: {summary['successful']}/{summary['total_files']}")
        print(f"❌ Failed: {summary['failed']}/{summary['total_files']}")
        print(f"📊 Success rate: {summary['success_rate']:.1%}")
    else:
        print(f"\n❌ Pipeline failed: {summary.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main() 