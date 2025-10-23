#!/usr/bin/env python3
"""
Test Display System

2-minute simulation of model training to test the display system.
Shows all stages, progress updates, and final summary.
"""

import time
import random
from display_manager import DisplayManager


def simulate_training():
    """Simulate 2-minute model training with all stages."""
    
    print("\n" + "="*80)
    print("DISPLAY SYSTEM TEST - 2 MINUTE SIMULATION")
    print("="*80)
    print("\nThis will simulate model training for 2 minutes to showcase")
    print("the progress display system with all stages and updates.")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize display manager
    dm = DisplayManager("Linear Regression", update_interval=20, enable_emojis=True)
    
    # Show training start
    total_stocks = 50  # Simulating 50 stocks
    dm.show_training_start(total_stocks=total_stocks, expected_duration_min=2)
    
    time.sleep(2)
    
    # Simulate 5 batches of 10 stocks each
    total_batches = 5
    stocks_per_batch = 10
    
    # US and Indian stock symbols for simulation
    us_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD', 
                 'NFLX', 'DIS', 'BA', 'GE', 'JPM', 'BAC', 'WMT', 'TGT', 'HD', 
                 'NKE', 'SBUX', 'MCD', 'V', 'MA', 'PYPL', 'INTC', 'CSCO']
    
    ind_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'WIPRO',
                  'ITC', 'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'MARUTI', 'TATAMOTORS',
                  'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'BAJFINANCE', 
                  'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'TECHM', 'HCLTECH', 
                  'POWERGRID', 'NTPC', 'ONGC']
    
    # Stages to cycle through
    stages = ['loading', 'validation', 'feature_engineering', 'preprocessing']
    
    start_time = time.time()
    stocks_processed = 0
    
    for batch_num in range(total_batches):
        # Determine batch composition (mix of US and Indian stocks)
        us_count = random.randint(6, 9)
        ind_count = stocks_per_batch - us_count
        
        # Select sample stocks for this batch
        batch_us_stocks = random.sample(us_stocks, us_count)
        batch_ind_stocks = random.sample(ind_stocks, ind_count)
        sample_stocks = batch_us_stocks + batch_ind_stocks
        random.shuffle(sample_stocks)
        
        # Show stage transition
        current_stage = stages[batch_num % len(stages)]
        stage_name = dm.STAGES[current_stage]['name']
        dm.show_stage_transition(batch_num + 1, total_batches, f"{stage_name} phase")
        
        time.sleep(1)
        
        # Simulate batch processing with multiple updates
        updates_per_batch = 2 if batch_num < total_batches - 1 else 3
        
        for update in range(updates_per_batch):
            stocks_processed += stocks_per_batch // updates_per_batch
            elapsed = time.time() - start_time
            
            # Calculate remaining time (aim for 2 minutes total)
            target_duration = 120  # 2 minutes
            progress = stocks_processed / total_stocks
            estimated_total = target_duration
            estimated_remaining = max(0, estimated_total - elapsed)
            
            # Build batch info
            batch_info = {
                'batch_num': batch_num,
                'total_batches': total_batches,
                'stocks_processed': stocks_processed,
                'total_stocks': total_stocks,
                'current_stage': current_stage,
                'us_stock_count': us_count,
                'ind_stock_count': ind_count,
                'sample_stocks': sample_stocks,
                'elapsed_time': elapsed,
                'estimated_remaining': estimated_remaining
            }
            
            # Show progress update
            dm.show_batch_progress(batch_info)
            
            # Sleep for a bit to simulate processing
            if batch_num < total_batches - 1 or update < updates_per_batch - 1:
                time.sleep(8)  # Sleep for 8 seconds between updates
        
        # Show batch completion
        if batch_num < total_batches - 1:
            dm.show_stage_transition(batch_num + 1, total_batches, 
                                    f"{stage_name} completed", stocks_per_batch)
            time.sleep(2)
    
    # Final training stage
    print("\n")
    dm.show_stage_transition(total_batches, total_batches, "Training phase")
    time.sleep(3)
    
    # Show model training progress
    batch_info = {
        'batch_num': total_batches - 1,
        'total_batches': total_batches,
        'stocks_processed': total_stocks,
        'total_stocks': total_stocks,
        'current_stage': 'training',
        'us_stock_count': 0,
        'ind_stock_count': 0,
        'sample_stocks': [],
        'elapsed_time': time.time() - start_time,
        'estimated_remaining': 5
    }
    dm.show_batch_progress(batch_info)
    
    time.sleep(5)
    
    # Show validation stage
    dm.show_stage_transition(total_batches, total_batches, "Validation phase")
    time.sleep(2)
    
    batch_info['current_stage'] = 'validation_final'
    batch_info['estimated_remaining'] = 0
    dm.show_batch_progress(batch_info)
    
    time.sleep(2)
    
    # Show completion summary
    total_time = time.time() - start_time
    
    summary = {
        'model_name': 'Linear Regression',
        'file_type': '.pkl',
        'model_path': 'backend/models/linear_regression/linear_regression_model.pkl',
        'file_size_mb': 2.3,
        'stocks_processed': total_stocks,
        'total_samples': total_stocks * 1247,  # Simulated sample count
        'total_time': total_time,
        'validation_r2': 0.9456
    }
    
    dm.show_training_complete(summary)
    
    # Final test summary
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print(f"\nEmoji Support: {'✓ Working' if dm.emojis_working else '✗ Not supported (text-only mode)'}")
    print(f"Total Simulation Time: {total_time:.1f} seconds (~{total_time/60:.1f} minutes)")
    print(f"Progress Updates Shown: {total_batches * 2 + 2}")
    print(f"\nAll display features tested:")
    print("  ✓ Training start display")
    print("  ✓ Batch progress updates with sample stocks")
    print("  ✓ Stage transitions")
    print("  ✓ Stock composition (US vs Indian)")
    print("  ✓ Progress bars (batch and stock)")
    print("  ✓ Time estimates")
    print("  ✓ Final completion summary with model details")
    print("\nIf the display looks good, you can proceed with actual training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    simulate_training()

