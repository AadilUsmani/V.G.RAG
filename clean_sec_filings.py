"""
Enhanced SEC 10-K Filing Cleaner
Cleans and extracts readable text from SEC EDGAR filings
"""

import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime


class SECFilingCleaner:
    """Cleans SEC filing documents and extracts readable text"""
    
    def __init__(self, source_dir="sec-edgar-filings", output_dir="clean_data"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_size_before': 0,
            'total_size_after': 0
        }
    
    def clean_text(self, file_path):
        """
        Clean a single SEC filing text file
        
        Args:
            file_path: Path to the filing file
            
        Returns:
            Cleaned text content or None if failed
        """
        try:
            # Read file with error handling for encoding issues
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                raw_content = f.read()
            
            original_size = len(raw_content)
            self.stats['total_size_before'] += original_size
            
            # Parse HTML/XML using BeautifulSoup
            soup = BeautifulSoup(raw_content, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator="\n")
            
            # SEC-specific cleaning
            text = self._clean_sec_formatting(text)
            
            # General text cleaning
            text = self._clean_general_text(text)
            
            self.stats['total_size_after'] += len(text)
            
            return text
            
        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
            return None
    
    def _clean_sec_formatting(self, text):
        """Remove SEC-specific formatting and headers"""
        
        # Remove SEC header information (common in EDGAR filings)
        text = re.sub(r'<SEC-DOCUMENT>.*?</SEC-DOCUMENT>', '', text, flags=re.DOTALL)
        text = re.sub(r'<SEC-HEADER>.*?</SEC-HEADER>', '', text, flags=re.DOTALL)
        
        # Remove XBRL tags
        text = re.sub(r'</?[A-Z][A-Z0-9_-]*:[A-Z][A-Z0-9_-]*[^>]*>', '', text)
        
        # Remove XML declarations
        text = re.sub(r'<\?xml[^>]*\?>', '', text)
        
        # Remove document type declarations
        text = re.sub(r'<!DOCTYPE[^>]*>', '', text)
        
        # Remove excessive dashes and underscores (table formatting artifacts)
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'={3,}', '', text)
        
        # Remove page numbers and headers that repeat
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_general_text(self, text):
        """General text cleaning operations"""
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        
        # Remove lines with only special characters
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip() and not re.match(r'^[^\w\s]+$', line.strip())]
        
        # Remove very short lines (likely artifacts) unless they're section numbers
        cleaned_lines = []
        for line in lines:
            if len(line) > 3 or re.match(r'^\d+\.?\s*$', line) or re.match(r'^[IVX]+\.?\s*$', line):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace cleanup
        text = text.strip()
        
        return text
    
    def _extract_metadata(self, file_path):
        """Extract company name and filing info from path"""
        path_parts = file_path.parts
        
        # Find company ticker (should be before the filing type folder)
        company = None
        filing_id = None
        
        for i, part in enumerate(path_parts):
            if part.upper() in ['AAPL', 'TSLA', 'APPLE', 'TESLA']:
                company = part.upper()
                if i + 1 < len(path_parts):
                    filing_id = path_parts[i + 2] if i + 2 < len(path_parts) else path_parts[i + 1]
                break
        
        # Fallback: use parent folder names
        if not company:
            company = path_parts[-3] if len(path_parts) >= 3 else "UNKNOWN"
            filing_id = path_parts[-2] if len(path_parts) >= 2 else "UNKNOWN"
        
        return company, filing_id
    
    def process_all_filings(self):
        """Process all SEC filing files in the source directory"""
        
        print("🧹 Starting SEC Filing Data Cleaning Process...")
        print(f"📂 Source: {self.source_dir.absolute()}")
        print(f"💾 Output: {self.output_dir.absolute()}\n")
        
        # Find all .txt files
        txt_files = list(self.source_dir.rglob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  No .txt files found in {self.source_dir}")
            return
        
        print(f"📄 Found {len(txt_files)} filing(s) to process\n")
        
        # Process each file
        for file_path in txt_files:
            self.stats['processed'] += 1
            
            # Extract metadata
            company, filing_id = self._extract_metadata(file_path)
            
            print(f"[{self.stats['processed']}/{len(txt_files)}] Processing: {company} - {filing_id}...")
            
            # Clean the content
            clean_content = self.clean_text(file_path)
            
            if clean_content:
                # Create meaningful filename
                timestamp = datetime.now().strftime("%Y%m%d")
                safe_name = f"{company}_{filing_id}_{timestamp}_cleaned.txt"
                save_path = self.output_dir / safe_name
                
                # Save cleaned file
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(clean_content)
                    
                    self.stats['successful'] += 1
                    file_size_kb = len(clean_content) / 1024
                    print(f"   ✅ Saved: {safe_name} ({file_size_kb:.1f} KB)")
                    
                except Exception as e:
                    print(f"   ❌ Failed to save: {e}")
                    self.stats['failed'] += 1
            else:
                self.stats['failed'] += 1
        
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary statistics"""
        print("\n" + "="*60)
        print("📊 CLEANING SUMMARY")
        print("="*60)
        print(f"Total files processed: {self.stats['processed']}")
        print(f"Successfully cleaned:  {self.stats['successful']} ✅")
        print(f"Failed:                {self.stats['failed']} ❌")
        
        if self.stats['total_size_before'] > 0:
            size_before_mb = self.stats['total_size_before'] / (1024 * 1024)
            size_after_mb = self.stats['total_size_after'] / (1024 * 1024)
            reduction = ((self.stats['total_size_before'] - self.stats['total_size_after']) / 
                        self.stats['total_size_before'] * 100)
            
            print(f"\nOriginal size:  {size_before_mb:.2f} MB")
            print(f"Cleaned size:   {size_after_mb:.2f} MB")
            print(f"Size reduction: {reduction:.1f}%")
        
        print(f"\n💾 Output directory: {self.output_dir.absolute()}")
        print("="*60)
        print("\n🎉 All files processed!")


def main():
    """Main execution function"""
    
    # You can customize these paths
    SOURCE_DIR = "sec-edgar-filings"
    OUTPUT_DIR = "clean_data"
    
    # Create cleaner instance and process files
    cleaner = SECFilingCleaner(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR)
    cleaner.process_all_filings()


if __name__ == "__main__":
    main()