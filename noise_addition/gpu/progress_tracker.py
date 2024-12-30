# progress_tracker.py
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, total_documents, description="Processing"):
        self.total = total_documents
        self.processed = 0
        self.start_time = datetime.now()
        self.success_count = 0
        self.error_count = 0
        self.current_doc_id = None
        self.pbar = tqdm(total=total_documents, desc=description)
        
    def update(self, doc_id, success=True, time_taken=None):
        self.processed += 1
        self.current_doc_id = doc_id
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
        self.pbar.update(1)
        
        # Calculate statistics
        elapsed = datetime.now() - self.start_time
        docs_per_second = self.processed / elapsed.total_seconds()
        remaining_docs = self.total - self.processed
        eta = timedelta(seconds=int(remaining_docs / docs_per_second)) if docs_per_second > 0 else "Unknown"
        
        # Log progress
        if time_taken:
            logger.info(
                f"Processed document {doc_id} in {time_taken:.2f}s | "
                f"Progress: {self.processed}/{self.total} ({self.processed/self.total*100:.1f}%) | "
                f"Success: {self.success_count} Errors: {self.error_count} | "
                f"ETA: {eta}"
            )

    def finalize(self):
        total_time = datetime.now() - self.start_time
        self.pbar.close()
        
        logger.info(
            f"\nProcessing completed in {total_time}\n"
            f"Total documents: {self.total}\n"
            f"Successful: {self.success_count}\n"
            f"Errors: {self.error_count}\n"
            f"Average speed: {self.processed/total_time.total_seconds():.2f} docs/second"
        )
