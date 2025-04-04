#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for downloading and processing BTCUSDT data.
This script combines the data download and feature engineering steps.
"""

import os
import argparse
import subprocess
import time
from datetime import datetime, timedelta
from utils import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='End-to-end data pipeline for BTCUSDT')
    
    parser.add_argument('--start_date', type=str, 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--end_date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format')
    
    parser.add_argument('--timeframes', type=str, nargs='+', 
                        default=['1h', '4h', '1d'],
                        help='List of timeframes to download')
    
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory to save raw data')
                        
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
                        
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of features')
                        
    parser.add_argument('--viz_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
                        
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol')
    
    return parser.parse_args()

def main():
    """Main function to run the complete data pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("data_pipeline", log_level="INFO")
    logger.info(f"Starting complete data pipeline for {args.symbol}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Ensure directories exist
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    # Track files for processing
    files_to_process = []
    
    # Download data for each timeframe
    for timeframe in args.timeframes:
        logger.info(f"Processing timeframe: {timeframe}")
        
        # Construct download command
        download_cmd = [
            "python", "download_btcusdt_data.py",
            "--start_date", args.start_date,
            "--end_date", args.end_date,
            "--timeframe", timeframe,
            "--output_dir", args.raw_dir,
            "--symbol", args.symbol
        ]
        
        # Run download command
        logger.info(f"Running: {' '.join(download_cmd)}")
        download_result = subprocess.run(download_cmd, capture_output=True, text=True)
        
        if download_result.returncode != 0:
            logger.error(f"Download failed for timeframe {timeframe}")
            logger.error(f"Error: {download_result.stderr}")
            continue
        
        logger.info(f"Download successful for timeframe {timeframe}")
        
        # Identify downloaded file
        filename = f"{args.symbol}_{timeframe}_{args.start_date}_{args.end_date}.csv"
        file_path = os.path.join(args.raw_dir, filename)
        
        if os.path.exists(file_path):
            files_to_process.append(file_path)
            logger.info(f"Added {file_path} to processing queue")
        else:
            logger.warning(f"Expected file {file_path} not found")
    
    # Process data files
    for file_path in files_to_process:
        logger.info(f"Processing file: {file_path}")
        
        # Construct processing command
        process_cmd = [
            "python", "process_btcusdt_data.py",
            "--input_file", file_path,
            "--output_dir", args.processed_dir
        ]
        
        if args.visualize:
            process_cmd.extend(["--visualize", "--viz_dir", args.viz_dir])
        
        # Run processing command
        logger.info(f"Running: {' '.join(process_cmd)}")
        process_result = subprocess.run(process_cmd, capture_output=True, text=True)
        
        if process_result.returncode != 0:
            logger.error(f"Processing failed for {file_path}")
            logger.error(f"Error: {process_result.stderr}")
        else:
            logger.info(f"Processing successful for {file_path}")
    
    # Generate summary report
    processed_files = [f for f in os.listdir(args.processed_dir) if f.endswith('_processed.csv')]
    
    logger.info("=== Pipeline Summary ===")
    logger.info(f"Total timeframes requested: {len(args.timeframes)}")
    logger.info(f"Files downloaded: {len(files_to_process)}")
    logger.info(f"Files processed: {len(processed_files)}")
    
    if len(processed_files) > 0:
        logger.info("Processed files:")
        for idx, filename in enumerate(processed_files, 1):
            file_path = os.path.join(args.processed_dir, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"{idx}. {filename} ({file_size:.2f} MB)")
    
    logger.info("Data pipeline completed")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Pipeline completed in {elapsed_time:.2f} seconds")