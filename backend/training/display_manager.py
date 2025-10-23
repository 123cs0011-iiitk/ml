#!/usr/bin/env python3
"""
Display Manager for Training Progress

Handles all display formatting with emoji support, stage tracking, and batch-level progress.
"""

import sys
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class DisplayManager:
    """Manages all training progress display with emoji support and stage tracking."""
    
    # Stage definitions with emojis
    STAGES = {
        'loading': {'name': 'Loading', 'emoji': '⏳', 'description': 'Reading CSV files from disk'},
        'validation': {'name': 'Validation', 'emoji': '✓', 'description': 'Checking data quality'},
        'feature_engineering': {'name': 'Feature Engineering', 'emoji': '🔧', 'description': 'Creating 37 technical indicators'},
        'preprocessing': {'name': 'Preprocessing', 'emoji': '📊', 'description': 'Cleaning and scaling data'},
        'training': {'name': 'Training', 'emoji': '🎯', 'description': 'Model fitting'},
        'validation_final': {'name': 'Validation', 'emoji': '✅', 'description': 'Testing model accuracy'}
    }
    
    def __init__(self, model_name: str, update_interval: int = 20, enable_emojis: bool = True):
        """
        Initialize display manager.
        
        Args:
            model_name: Name of the model being trained
            update_interval: Update interval in seconds
            enable_emojis: Whether to try using emojis (auto-fallback if errors)
        """
        self.model_name = model_name.upper().replace('_', ' ')
        self.update_interval = update_interval
        self.enable_emojis = enable_emojis
        self.emojis_working = enable_emojis
        
        # Test emoji support
        if self.enable_emojis:
            self._test_emoji_support()
        
        # Separators
        self.separator_double = '═' * 80
        self.separator_single = '─' * 80
        
    def _test_emoji_support(self):
        """Test if terminal supports emojis and disable if not."""
        try:
            # Try to encode some common emojis
            test_emojis = '🎯📊🔧✓✅⏳'
            test_emojis.encode(sys.stdout.encoding or 'utf-8')
            self.emojis_working = True
        except (UnicodeEncodeError, AttributeError, TypeError):
            self.emojis_working = False
            print("Note: Terminal doesn't support emojis, using text-only display")
    
    def _get_emoji(self, stage_key: str) -> str:
        """Get emoji for stage, or empty string if emojis disabled."""
        if not self.emojis_working:
            return ''
        return self.STAGES.get(stage_key, {}).get('emoji', '')
    
    def _safe_print(self, text: str):
        """Safely print text, catching emoji errors."""
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback: remove emojis and try again
            self.emojis_working = False
            # Simple emoji removal (remove all non-ASCII)
            ascii_text = ''.join(char for char in text if ord(char) < 128)
            print(ascii_text)
    
    def create_progress_bar(self, percentage: float, length: int = 50) -> str:
        """Create a progress bar string."""
        filled = int(length * percentage / 100)
        bar = '█' * filled + '─' * (length - filled)
        return f"[{bar}]"
    
    def format_time(self, seconds: float) -> str:
        """Format time in a readable way."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hr"
    
    def show_training_start(self, total_stocks: int, expected_duration_min: int):
        """Show training start message."""
        emoji_start = '🚀' if self.emojis_working else '>>>'
        emoji_data = '📊' if self.emojis_working else '[DATA]'
        emoji_time = '⏱️' if self.emojis_working else '[TIME]'
        emoji_model = '🎯' if self.emojis_working else '[MODEL]'
        
        self._safe_print(f"\n{self.separator_double}")
        self._safe_print(f"{emoji_start} STARTING {self.model_name} TRAINING")
        self._safe_print(f"{self.separator_single}")
        self._safe_print(f"{emoji_data} DATA TARGET:")
        self._safe_print(f"   Stocks: ~{total_stocks:,} stocks (US + Indian)")
        self._safe_print(f"   Historical Data: 5 years per stock")
        self._safe_print(f"   Features: 37 technical indicators per stock")
        self._safe_print(f"   Total Samples: ~{total_stocks * 1000:,}+ data points")
        self._safe_print(f"")
        self._safe_print(f"{emoji_time} TIMING ESTIMATES:")
        self._safe_print(f"   Start Time: {datetime.now().strftime('%H:%M:%S')}")
        self._safe_print(f"   Expected Duration: ~{expected_duration_min} minutes")
        self._safe_print(f"   Updates: Every {self.update_interval} seconds")
        self._safe_print(f"")
        self._safe_print(f"{emoji_model} MODEL INFO:")
        self._safe_print(f"   Model: {self.model_name}")
        self._safe_print(f"   Progress tracking: Batch-level updates")
        self._safe_print(f"{self.separator_double}\n")
    
    def show_batch_progress(self, batch_info: Dict[str, Any]):
        """
        Show batch-level progress update.
        
        Args:
            batch_info: Dictionary with:
                - batch_num: Current batch number (0-indexed)
                - total_batches: Total number of batches
                - stocks_processed: Number of stocks processed so far
                - total_stocks: Total stocks to process
                - current_stage: Current stage key
                - us_stock_count: Number of US stocks in batch
                - ind_stock_count: Number of Indian stocks in batch
                - sample_stocks: List of sample stock symbols
                - elapsed_time: Elapsed time in seconds
                - estimated_remaining: Estimated remaining time in seconds
        """
        batch_num = batch_info.get('batch_num', 0) + 1  # Convert to 1-indexed
        total_batches = batch_info.get('total_batches', 1)
        stocks_processed = batch_info.get('stocks_processed', 0)
        total_stocks = batch_info.get('total_stocks', 1000)
        current_stage = batch_info.get('current_stage', 'loading')
        us_count = batch_info.get('us_stock_count', 0)
        ind_count = batch_info.get('ind_stock_count', 0)
        sample_stocks = batch_info.get('sample_stocks', [])
        elapsed = batch_info.get('elapsed_time', 0)
        remaining = batch_info.get('estimated_remaining', 0)
        
        # Calculate percentages
        batch_percent = (batch_num / total_batches) * 100
        stock_percent = (stocks_processed / total_stocks) * 100 if total_stocks > 0 else 0
        
        # Get stage info
        stage_info = self.STAGES.get(current_stage, self.STAGES['loading'])
        stage_emoji = self._get_emoji(current_stage)
        stage_name = stage_info['name']
        stage_desc = stage_info['description']
        
        # Emojis
        emoji_batch = '📊' if self.emojis_working else '[BATCH]'
        emoji_stock = '📈' if self.emojis_working else '[STOCK]'
        emoji_us = '🇺🇸' if self.emojis_working else '[US]'
        emoji_ind = '🇮🇳' if self.emojis_working else '[IND]'
        emoji_time = '⏱️' if self.emojis_working else '[TIME]'
        emoji_target = '🎯' if self.emojis_working else '[ETA]'
        emoji_location = '📍' if self.emojis_working else '>'
        emoji_list = '📝' if self.emojis_working else '-'
        emoji_gear = '⚙️' if self.emojis_working else '[>]'
        
        # Format sample stocks (first 8)
        sample_display = ', '.join(sample_stocks[:8])
        if len(sample_stocks) > 8:
            sample_display += '...'
        
        # Calculate rate
        rate = stocks_processed / (elapsed / 60) if elapsed > 0 else 0
        
        # Estimated completion time
        if remaining > 0:
            completion_time = time.time() + remaining
            completion_str = time.strftime("%H:%M:%S", time.localtime(completion_time))
        else:
            completion_str = "Calculating..."
        
        # Build and print display
        self._safe_print(f"\n{self.separator_double}")
        self._safe_print(f"{stage_emoji} {self.model_name} - TRAINING IN PROGRESS")
        self._safe_print(f"{self.separator_single}")
        
        # Progress bars
        batch_bar = self.create_progress_bar(batch_percent)
        self._safe_print(f"{emoji_batch} Batch Progress:   {batch_bar} {batch_percent:.1f}% ({batch_num}/{total_batches} batches)")
        
        stock_bar = self.create_progress_bar(stock_percent)
        self._safe_print(f"{emoji_stock} Stock Progress:   {stock_bar} {stock_percent:.1f}% ({stocks_processed:,}/{total_stocks:,} stocks)")
        
        self._safe_print(f"")
        self._safe_print(f"Current Stage: {stage_name} {stage_emoji}")
        self._safe_print(f"Current Batch: Batch {batch_num}/{total_batches}")
        
        if us_count > 0 or ind_count > 0:
            self._safe_print(f"  {emoji_location} Composition: {us_count} US stocks {emoji_us} | {ind_count} Indian stocks {emoji_ind}")
        
        if sample_display:
            self._safe_print(f"  {emoji_list} Sample Stocks: {sample_display}")
        
        self._safe_print(f"  {emoji_gear}  Processing: {stage_desc}")
        
        self._safe_print(f"")
        self._safe_print(f"{emoji_time}  Elapsed: {self.format_time(elapsed)} | Remaining: ~{self.format_time(remaining)} | Rate: {rate:.1f} stocks/min")
        self._safe_print(f"{emoji_target} Expected Completion: {completion_str}")
        self._safe_print(f"{self.separator_double}\n")
    
    def show_stage_transition(self, batch_num: int, total_batches: int, 
                             stage_name: str, stocks_in_batch: int = 0):
        """Show a quick stage transition notification."""
        emoji_check = '✓' if self.emojis_working else '[OK]'
        emoji_next = '→' if self.emojis_working else '>>'
        
        batch_display = f"Batch {batch_num}/{total_batches}"
        
        if 'completed' in stage_name.lower() or 'finished' in stage_name.lower():
            msg = f"{emoji_check} {batch_display} completed - {stage_name}"
            if stocks_in_batch > 0:
                msg += f" ({stocks_in_batch} stocks processed)"
        else:
            msg = f"{emoji_next} {batch_display} starting - {stage_name}..."
        
        self._safe_print(msg)
    
    def show_training_complete(self, summary: Dict[str, Any]):
        """
        Show final training completion summary.
        
        Args:
            summary: Dictionary with:
                - model_name: Full model name
                - file_type: File extension (.pkl, .h5, etc.)
                - model_path: Full path to saved model
                - file_size_mb: File size in MB
                - stocks_processed: Total stocks processed
                - total_samples: Total samples used
                - total_time: Total time in seconds
                - validation_r2: Validation R² score
        """
        emoji_success = '✅' if self.emojis_working else '[SUCCESS]'
        emoji_model = '📦' if self.emojis_working else '[MODEL]'
        emoji_file = '💾' if self.emojis_working else '[FILE]'
        emoji_folder = '📁' if self.emojis_working else '[PATH]'
        emoji_size = '📊' if self.emojis_working else '[SIZE]'
        emoji_stocks = '📈' if self.emojis_working else '[DATA]'
        emoji_samples = '📊' if self.emojis_working else '[SAMPLES]'
        emoji_time = '⏱️' if self.emojis_working else '[TIME]'
        emoji_score = '🎯' if self.emojis_working else '[SCORE]'
        emoji_status = '✅' if self.emojis_working else '[OK]'
        emoji_check = '✓' if self.emojis_working else 'OK'
        
        model_name = summary.get('model_name', self.model_name)
        file_type = summary.get('file_type', '.pkl')
        model_path = summary.get('model_path', '')
        file_size = summary.get('file_size_mb', 0)
        stocks = summary.get('stocks_processed', 0)
        samples = summary.get('total_samples', 0)
        total_time = summary.get('total_time', 0)
        r2_score = summary.get('validation_r2', 0)
        
        self._safe_print(f"\n{self.separator_double}")
        self._safe_print(f"{emoji_success} TRAINING COMPLETED SUCCESSFULLY")
        self._safe_print(f"{self.separator_double}")
        self._safe_print(f"Model Details:")
        self._safe_print(f"  {emoji_model} Model Name: {model_name}")
        self._safe_print(f"  {emoji_file} File Type: {file_type} (Pickle)" if file_type == '.pkl' else f"  {emoji_file} File Type: {file_type}")
        self._safe_print(f"  {emoji_folder} Saved Path: {model_path}")
        if file_size > 0:
            self._safe_print(f"  {emoji_size} File Size: {file_size:.1f} MB")
        self._safe_print(f"")
        self._safe_print(f"Training Summary:")
        self._safe_print(f"  {emoji_stocks} Stocks Processed: {stocks:,} / {stocks:,} (100%)")
        self._safe_print(f"  {emoji_samples} Total Samples: {samples:,}")
        self._safe_print(f"  {emoji_time}  Total Time: {self.format_time(total_time)}")
        if r2_score > 0:
            self._safe_print(f"  {emoji_score} Validation R²: {r2_score:.4f}")
        self._safe_print(f"  {emoji_status} Status: Model ready for predictions")
        self._safe_print(f"")
        self._safe_print(f"Model saved successfully {emoji_check}")
        self._safe_print(f"{self.separator_double}\n")
    
    def show_error(self, error_message: str, batch_num: Optional[int] = None):
        """Show error message."""
        emoji_error = '❌' if self.emojis_working else '[ERROR]'
        
        self._safe_print(f"\n{self.separator_single}")
        if batch_num is not None:
            self._safe_print(f"{emoji_error} Error in Batch {batch_num}: {error_message}")
        else:
            self._safe_print(f"{emoji_error} Error: {error_message}")
        self._safe_print(f"{self.separator_single}\n")


# Example usage
if __name__ == "__main__":
    # Test the display manager
    dm = DisplayManager("Linear Regression")
    
    print("Testing emoji support...")
    print(f"Emojis working: {dm.emojis_working}")
    
    print("\n\nTesting training start display...")
    dm.show_training_start(total_stocks=1000, expected_duration_min=25)
    
    print("\n\nTesting batch progress display...")
    batch_info = {
        'batch_num': 5,
        'total_batches': 10,
        'stocks_processed': 350,
        'total_stocks': 1000,
        'current_stage': 'feature_engineering',
        'us_stock_count': 85,
        'ind_stock_count': 15,
        'sample_stocks': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'RELIANCE', 'TCS', 'INFY', 'WIPRO'],
        'elapsed_time': 930,  # 15.5 minutes
        'estimated_remaining': 612  # 10.2 minutes
    }
    dm.show_batch_progress(batch_info)
    
    print("\n\nTesting stage transition...")
    dm.show_stage_transition(5, 10, "Data loading finished", 100)
    
    print("\n\nTesting completion summary...")
    summary = {
        'model_name': 'Linear Regression',
        'file_type': '.pkl',
        'model_path': 'backend/models/linear_regression/linear_regression_model.pkl',
        'file_size_mb': 2.3,
        'stocks_processed': 1000,
        'total_samples': 1245678,
        'total_time': 1530,  # 25.5 minutes
        'validation_r2': 0.9456
    }
    dm.show_training_complete(summary)

