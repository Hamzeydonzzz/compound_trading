"""
main.py - Main execution module for Project Compound Trading Bot

This module serves as the central coordinator for the entire trading system,
integrating all components including:
- Data acquisition and processing
- Model inference
- Trading logic
- Risk management
- Execution via Binance API
- Monitoring and reporting

The main loop handles the continuous operation of the bot, error recovery,
and system shutdown procedures.
"""

import os
import sys
import time
import signal
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Optional, Union, Tuple

# Import project modules
from config import Config
from logger import setup_logger
from data_handler import DataHandler
from binance_interface import BinanceInterface
from feature_engineering import FeatureEngineering
from inference import InferenceEngine, RealTimeInference
from trading_logic import TradingLogic, Trade, Position
import utils

# Global variables
running = True  # Control flag for main loop
config = None
logger = None
data_queue = queue.Queue(maxsize=100)  # Queue for data updates
signal_queue = queue.Queue(maxsize=100)  # Queue for trading signals
error_queue = queue.Queue(maxsize=100)  # Queue for error handling


class TradingBot:
    """
    Main trading bot class coordinating all components
    """
    
    def __init__(self, config_path: str, log_level: str = 'INFO'):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Setup logging
        self.logger = setup_logger(__name__, log_level)
        
        # Initialize components
        self.data_handler = None
        self.binance_interface = None
        self.inference_engine = None
        self.real_time_inference = None
        self.trading_logic = None
        self.feature_engineering = None
        
        # Trading state
        self.active_trade = None
        self.account_info = {}
        self.current_position = Position.FLAT
        self.last_update_time = None
        self.last_signal_time = None
        
        # Stats and metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_fees': 0.0,
            'max_drawdown': 0.0
        }
        
        # Control flags
        self.running = True
        self.initialized = False
        self.in_error_state = False
        self.recovery_attempts = 0
        
        # Threads and queues
        self.data_thread = None
        self.inference_thread = None
        self.trading_thread = None
        self.monitoring_thread = None
        
        self.data_queue = queue.Queue(maxsize=100)
        self.signal_queue = queue.Queue(maxsize=100)
        self.error_queue = queue.Queue(maxsize=100)
        
        # Initialize system
        self._initialize()
        
    def _initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            True if initialization is successful, False otherwise
        """
        try:
            self.logger.info("Initializing trading bot...")
            
            # Initialize data handler
            self.logger.info("Initializing data handler...")
            self.data_handler = DataHandler(self.config)
            
            # Initialize Binance interface
            self.logger.info("Initializing Binance interface...")
            self.binance_interface = BinanceInterface(self.config)
            
            # Check API connection
            if not self.binance_interface.check_connection():
                self.logger.error("Could not connect to Binance API.")
                return False
                
            # Get account info
            self.account_info = self.binance_interface.get_account_info()
            self.logger.info(f"Connected to Binance API. Account status: {self.account_info['status']}")
            
            # Initialize feature engineering
            self.logger.info("Initializing feature engineering...")
            self.feature_engineering = FeatureEngineering(self.config)
            
            # Initialize inference engine
            self.logger.info("Initializing inference engine...")
            self.inference_engine = InferenceEngine(self.config)
            
            # Initialize real-time inference
            self.logger.info("Initializing real-time inference...")
            self.real_time_inference = RealTimeInference(self.config, self.inference_engine)
            
            # Initialize trading logic
            self.logger.info("Initializing trading logic...")
            self.trading_logic = TradingLogic(self.config)
            
            # Set initialization state
            self.initialized = True
            self.in_error_state = False
            self.recovery_attempts = 0
            
            self.logger.info("Trading bot initialized successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.in_error_state = True
            return False
            
    def start(self):
        """Start the trading bot"""
        if not self.initialized:
            if not self._initialize():
                self.logger.error("Could not initialize trading bot. Exiting.")
                return
                
        self.logger.info("Starting trading bot...")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start threads
        self._start_threads()
        
        # Main control loop
        try:
            while self.running:
                # Process error queue
                self._process_errors()
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error in main control loop: {e}")
            self.running = False
            
        finally:
            # Cleanup and shutdown
            self._shutdown()
            
    def _start_threads(self):
        """Start the worker threads"""
        # Data collection thread
        self.data_thread = threading.Thread(
            target=self._data_worker,
            name="DataThread",
            daemon=True
        )
        self.data_thread.start()
        
        # Inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            name="InferenceThread",
            daemon=True
        )
        self.inference_thread.start()
        
        # Trading logic thread
        self.trading_thread = threading.Thread(
            target=self._trading_worker,
            name="TradingThread",
            daemon=True
        )
        self.trading_thread.start()
        
        # Monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="MonitoringThread",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("All worker threads started")
        
    def _data_worker(self):
        """Worker thread for data collection"""
        self.logger.info("Data worker thread started")
        
        interval_seconds = self.config.get('bot.data_update_interval', 60)
        symbol = self.config.get('data.symbol', 'BTCUSDT')
        timeframe = self.config.get('data.timeframe', '1h')
        limit = self.config.get('data.limit', 500)
        
        while self.running:
            try:
                # Fetch latest data
                self.logger.debug(f"Fetching latest data for {symbol} ({timeframe})")
                data = self.data_handler.fetch_latest_data(symbol, timeframe, limit)
                
                # Add technical indicators
                data = self.feature_engineering.add_all_features(data)
                
                # Put in queue for other threads
                if not self.data_queue.full():
                    self.data_queue.put({
                        'time': datetime.now(),
                        'data': data,
                        'symbol': symbol,
                        'timeframe': timeframe
                    })
                    self.last_update_time = datetime.now()
                else:
                    self.logger.warning("Data queue is full. Skipping update.")
                
            except Exception as e:
                self.logger.error(f"Error in data worker: {e}")
                if not self.error_queue.full():
                    self.error_queue.put({
                        'time': datetime.now(),
                        'source': 'data_worker',
                        'error': str(e),
                        'recoverable': True
                    })
                
            # Sleep until next update
            time.sleep(interval_seconds)
            
    def _inference_worker(self):
        """Worker thread for model inference"""
        self.logger.info("Inference worker thread started")
        
        while self.running:
            try:
                # Get latest data from queue
                if not self.data_queue.empty():
                    data_item = self.data_queue.get()
                    
                    # Run inference
                    self.logger.debug("Running inference on latest data")
                    inference_result = self.real_time_inference.update(data_item['data'])
                    
                    # Put result in signal queue
                    if not self.signal_queue.full():
                        self.signal_queue.put({
                            'time': datetime.now(),
                            'inference_result': inference_result,
                            'data': data_item['data']
                        })
                    else:
                        self.logger.warning("Signal queue is full. Skipping inference result.")
                        
                    # Mark data task as done
                    self.data_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in inference worker: {e}")
                if not self.error_queue.full():
                    self.error_queue.put({
                        'time': datetime.now(),
                        'source': 'inference_worker',
                        'error': str(e),
                        'recoverable': True
                    })
                    
            # Short sleep to avoid high CPU usage
            time.sleep(0.1)
            
    def _trading_worker(self):
        """Worker thread for trading logic and execution"""
        self.logger.info("Trading worker thread started")
        
        while self.running:
            try:
                # Get latest signal from queue
                if not self.signal_queue.empty():
                    signal_item = self.signal_queue.get()
                    
                    # Process trading signal
                    self.logger.debug("Processing trading signal")
                    self._process_trading_signal(signal_item)
                    
                    # Mark signal task as done
                    self.signal_queue.task_done()
                    
                # Check for exit conditions on active trade
                if self.active_trade:
                    self._check_exit_conditions()
                    
            except Exception as e:
                self.logger.error(f"Error in trading worker: {e}")
                if not self.error_queue.full():
                    self.error_queue.put({
                        'time': datetime.now(),
                        'source': 'trading_worker',
                        'error': str(e),
                        'recoverable': True
                    })
                    
            # Short sleep to avoid high CPU usage
            time.sleep(0.1)
            
    def _monitoring_worker(self):
        """Worker thread for system monitoring and reporting"""
        self.logger.info("Monitoring worker thread started")
        
        # Set monitoring interval
        interval_seconds = self.config.get('bot.monitoring_interval', 300)  # 5 minutes
        
        while self.running:
            try:
                # Update account information
                self.account_info = self.binance_interface.get_account_info()
                
                # Generate status report
                self._generate_status_report()
                
                # Check for system health issues
                if self._check_system_health():
                    # Log that system is healthy
                    pass
                
            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                if not self.error_queue.full():
                    self.error_queue.put({
                        'time': datetime.now(),
                        'source': 'monitoring_worker',
                        'error': str(e),
                        'recoverable': True
                    })
                    
            # Sleep until next monitoring cycle
            time.sleep(interval_seconds)
            
    def _process_trading_signal(self, signal_item: Dict):
        """
        Process a trading signal
        
        Args:
            signal_item: Dictionary containing the signal information
        """
        inference_result = signal_item['inference_result']
        data = signal_item['data']
        
        # Skip if no inference result
        if not inference_result.get('success', False):
            self.logger.warning("Skipping invalid inference result")
            return
            
        # Get signal information
        signal = inference_result.get('signal', 'NONE')
        confidence = inference_result.get('confidence', 0.0)
        
        # Log the signal
        self.logger.info(f"Signal: {signal}, Confidence: {confidence:.4f}")
        
        # Skip if confidence is too low
        min_confidence = self.config.get('trading.confidence_threshold', 0.7)
        if confidence < min_confidence:
            self.logger.info(f"Signal confidence ({confidence:.4f}) below threshold ({min_confidence}). No action taken.")
            return
            
        # Process signal based on current position
        if self.current_position == Position.FLAT:
            # Only consider BUY signals when flat
            if signal == 'BUY':
                self._execute_entry(inference_result, data)
        elif self.current_position == Position.LONG:
            # Consider exit signals when in position
            if signal == 'SELL':
                self._execute_exit(inference_result, data, "Sell signal")
                
        # Update last signal time
        self.last_signal_time = datetime.now()
        
    def _execute_entry(self, inference_result: Dict, data: pd.DataFrame):
        """
        Execute an entry order
        
        Args:
            inference_result: Dictionary containing inference results
            data: DataFrame containing market data
        """
        # Skip if already in a position
        if self.active_trade:
            self.logger.warning("Already in a position. Skipping entry.")
            return
            
        # Get current market data
        latest_price = data['close'].iloc[-1]
        symbol = self.config.get('data.symbol', 'BTCUSDT')
        
        # Calculate position size (100% of available balance)
        balance = self._get_available_balance()
        position_size = balance
        
        # Calculate risk parameters
        stop_loss_pct = self.config.get('trading.stop_loss_percentage', 0.03)
        take_profit_pct = self.config.get('trading.take_profit_percentage', 0.06)
        
        stop_loss_price = latest_price * (1 - stop_loss_pct)
        take_profit_price = latest_price * (1 + take_profit_pct)
        
        # Execute the order
        self.logger.info(f"Executing entry: {symbol} at {latest_price:.2f}")
        
        try:
            # Create order
            order_result = self.binance_interface.create_order(
                symbol=symbol,
                side='BUY',
                quantity=position_size / latest_price,
                price=None  # Market order
            )
            
            # Check if order was successful
            if order_result.get('status') == 'FILLED':
                # Create trade object
                self.active_trade = Trade(
                    symbol=symbol,
                    entry_price=float(order_result.get('price', latest_price)),
                    entry_time=datetime.now(),
                    position_size=position_size,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    max_holding_time=self.config.get('trading.max_holding_time', 48)
                )
                
                # Update position state
                self.current_position = Position.LONG
                
                # Log the entry
                self.logger.info(
                    f"Entry executed: {symbol} at {self.active_trade.entry_price:.2f}, "
                    f"Size: {position_size:.2f}, "
                    f"Stop: {stop_loss_price:.2f}, "
                    f"Target: {take_profit_price:.2f}"
                )
                
                # Update metrics
                self.metrics['total_trades'] += 1
                
            else:
                self.logger.error(f"Order failed: {order_result}")
                
        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            if not self.error_queue.full():
                self.error_queue.put({
                    'time': datetime.now(),
                    'source': 'execute_entry',
                    'error': str(e),
                    'recoverable': True
                })
                
    def _execute_exit(self, inference_result: Dict, data: pd.DataFrame, reason: str):
        """
        Execute an exit order
        
        Args:
            inference_result: Dictionary containing inference results
            data: DataFrame containing market data
            reason: Reason for the exit
        """
        # Skip if not in a position
        if not self.active_trade:
            self.logger.warning("No active position. Skipping exit.")
            return
            
        # Get current market data
        latest_price = data['close'].iloc[-1]
        symbol = self.active_trade.symbol
        
        # Execute the order
        self.logger.info(f"Executing exit: {symbol} at {latest_price:.2f}, Reason: {reason}")
        
        try:
            # Create order
            order_result = self.binance_interface.create_order(
                symbol=symbol,
                side='SELL',
                quantity=self.active_trade.quantity,
                price=None  # Market order
            )
            
            # Check if order was successful
            if order_result.get('status') == 'FILLED':
                # Close the trade
                exit_price = float(order_result.get('price', latest_price))
                self.active_trade.close_trade(exit_price, datetime.now(), reason)
                
                # Update position state
                self.current_position = Position.FLAT
                
                # Calculate profit/loss
                pnl = self.active_trade.pnl
                pnl_pct = self.active_trade.pnl_percent
                
                # Log the exit
                self.logger.info(
                    f"Exit executed: {symbol} at {exit_price:.2f}, "
                    f"P/L: {pnl:.2f} ({pnl_pct:.2f}%), "
                    f"Reason: {reason}"
                )
                
                # Update metrics
                if pnl > 0:
                    self.metrics['winning_trades'] += 1
                else:
                    self.metrics['losing_trades'] += 1
                    
                self.metrics['total_profit'] += pnl
                
                # Save the trade to history
                self._save_trade_history(self.active_trade)
                
                # Clear active trade
                self.active_trade = None
                
            else:
                self.logger.error(f"Order failed: {order_result}")
                
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            if not self.error_queue.full():
                self.error_queue.put({
                    'time': datetime.now(),
                    'source': 'execute_exit',
                    'error': str(e),
                    'recoverable': True
                })
                
    def _check_exit_conditions(self):
        """Check for exit conditions on the active trade"""
        if not self.active_trade:
            return
            
        # Skip if no recent data
        if self.last_update_time is None:
            return
            
        # Get latest data from data queue
        latest_data = None
        for i in range(self.data_queue.qsize()):
            item = self.data_queue.get()
            latest_data = item
            self.data_queue.put(item)  # Put back in queue
            
        if latest_data is None:
            return
            
        data = latest_data['data']
        latest_price = data['close'].iloc[-1]
        
        # Update trade with current price
        self.active_trade.update_with_new_price(latest_price)
        
        # Check exit conditions
        exit_reason = None
        
        # Check if stop loss hit
        if latest_price <= self.active_trade.stop_loss:
            exit_reason = "Stop loss hit"
            
        # Check if take profit hit
        elif latest_price >= self.active_trade.take_profit:
            exit_reason = "Take profit hit"
            
        # Check if dynamic take profit hit
        elif (self.active_trade.dynamic_take_profit is not None and
              latest_price <= self.active_trade.dynamic_take_profit and
              latest_price > self.active_trade.entry_price):
            exit_reason = "Dynamic take profit hit"
            
        # Check if max holding time reached
        elif self.active_trade.should_exit_by_time():
            exit_reason = "Maximum holding time reached"
            
        # Execute exit if needed
        if exit_reason:
            self._execute_exit(None, data, exit_reason)
            
    def _process_errors(self):
        """Process errors from the error queue"""
        while not self.error_queue.empty():
            error_item = self.error_queue.get()
            
            # Log the error
            self.logger.error(
                f"Processing error from {error_item['source']}: {error_item['error']}"
            )
            
            # Handle different types of errors
            if error_item.get('recoverable', False):
                self._attempt_recovery(error_item)
            else:
                # Non-recoverable error - initiate shutdown
                self.logger.critical(f"Non-recoverable error: {error_item['error']}")
                self.running = False
                break
                
            # Mark error as processed
            self.error_queue.task_done()
            
    def _attempt_recovery(self, error_item: Dict):
        """
        Attempt to recover from an error
        
        Args:
            error_item: Dictionary containing error information
        """
        # Increment recovery attempts
        self.recovery_attempts += 1
        
        # Log recovery attempt
        self.logger.info(
            f"Recovery attempt {self.recovery_attempts} for error in {error_item['source']}"
        )
        
        # Check if max recovery attempts reached
        max_attempts = self.config.get('bot.max_recovery_attempts', 3)
        if self.recovery_attempts > max_attempts:
            self.logger.critical(
                f"Max recovery attempts ({max_attempts}) reached. Initiating shutdown."
            )
            self.running = False
            return
            
        # Attempt different recovery strategies based on the error source
        if error_item['source'] == 'data_worker':
            # For data errors, try to reinitialize the data handler
            try:
                self.data_handler = DataHandler(self.config)
                self.logger.info("Data handler reinitialized successfully")
                self.recovery_attempts = 0  # Reset on success
            except Exception as e:
                self.logger.error(f"Failed to reinitialize data handler: {e}")
                
        elif error_item['source'] == 'inference_worker':
            # For inference errors, try to reinitialize the inference engine
            try:
                self.inference_engine = InferenceEngine(self.config)
                self.real_time_inference = RealTimeInference(self.config, self.inference_engine)
                self.logger.info("Inference engine reinitialized successfully")
                self.recovery_attempts = 0  # Reset on success
            except Exception as e:
                self.logger.error(f"Failed to reinitialize inference engine: {e}")
                
        elif error_item['source'] in ['trading_worker', 'execute_entry', 'execute_exit']:
            # For trading errors, try to reinitialize the Binance interface
            try:
                self.binance_interface = BinanceInterface(self.config)
                self.logger.info("Binance interface reinitialized successfully")
                
                # Check if we need to recover an active trade
                if self.active_trade and self.current_position == Position.LONG:
                    # Check if the position still exists
                    position_info = self.binance_interface.get_position_info(self.active_trade.symbol)
                    
                    if position_info and float(position_info.get('positionAmt', 0)) > 0:
                        self.logger.info("Active position still exists. No recovery needed.")
                    else:
                        self.logger.warning("Active position not found. Resetting position state.")
                        self.active_trade = None
                        self.current_position = Position.FLAT
                        
                self.recovery_attempts = 0  # Reset on success
            except Exception as e:
                self.logger.error(f"Failed to reinitialize Binance interface: {e}")
        
    def _check_system_health(self) -> bool:
        """
        Check overall system health
        
        Returns:
            True if system is healthy, False otherwise
        """
        # Check if all threads are alive
        threads = [
            self.data_thread,
            self.inference_thread,
            self.trading_thread,
            self.monitoring_thread
        ]
        
        for thread in threads:
            if thread is not None and not thread.is_alive():
                self.logger.error(f"Thread {thread.name} is not running")
                return False
                
        # Check if data is being updated
        if self.last_update_time:
            time_since_update = (datetime.now() - self.last_update_time).total_seconds() / 60
            max_update_interval = self.config.get('bot.max_data_age_minutes', 15)
            
            if time_since_update > max_update_interval:
                self.logger.warning(
                    f"Data age ({time_since_update:.1f} min) exceeds maximum ({max_update_interval} min)"
                )
                return False
                
        # Check account status
        if self.account_info.get('status') != 'NORMAL':
            self.logger.warning(f"Account status is not normal: {self.account_info.get('status')}")
            return False
            
        return True
        
    def _generate_status_report(self):
        """Generate and log a status report"""
        # Get account information
        balances = self.account_info.get('balances', [])
        usdt_balance = 0
        
        for balance in balances:
            if balance['asset'] == 'USDT':
                usdt_balance = float(balance['free']) + float(balance['locked'])
                
        # Create status report
        report = {
            'time': datetime.now().isoformat(),
            'system_status': 'RUNNING' if self.running else 'STOPPING',
            'active_position': self.current_position.name,
            'metrics': self.metrics,
            'account': {
                'status': self.account_info.get('status'),
                'usdt_balance': usdt_balance,
                'active_trade': self.active_trade.to_dict() if self.active_trade else None
            },
            'performance': {
                'win_rate': self.metrics['winning_trades'] / self.metrics['total_trades'] 
                            if self.metrics['total_trades'] > 0 else 0,
                'total_profit': self.metrics['total_profit'],
                'max_drawdown': self.metrics['max_drawdown']
            }
        }
        
        # Log the report
        self.logger.info(f"Status report: {json.dumps(report, indent=2)}")
        
        # Save to file
        self._save_status_report(report)
        
    def _save_status_report(self, report: Dict):
        """
        Save the status report to a file
        
        Args:
            report: Dictionary containing status report
        """
        try:
            # Create reports directory if it doesn't exist
            reports_dir = self.config.get('bot.reports_dir', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Create filename based on timestamp
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = os.path.join(reports_dir, f"status-{timestamp}.json")
            
            # Save the report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving status report: {e}")
            
    def _save_trade_history(self, trade: Trade):
        """
        Save a completed trade to the trade history
        
        Args:
            trade: Completed trade object
        """
        try:
            # Create trade history directory if it doesn't exist
            history_dir = self.config.get('bot.trade_history_dir', 'trade_history')
            os.makedirs(history_dir, exist_ok=True)
            
            # Create filename based on trade ID
            filename = os.path.join(history_dir, f"trade-{trade.trade_id}.json")
            
            # Save the trade
            with open(filename, 'w') as f:
                json.dump(trade.to_dict(), f, indent=2)
                
            # Append to master trade log
            trade_log_path = os.path.join(history_dir, "trade_log.csv")
            
            # Create trade row
            trade_data = {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason
            }
            
            # Create DataFrame and save
            df = pd.DataFrame([trade_data])
            
            # Check if file exists
            if os.path.exists(trade_log_path):
                # Append to existing file
                df.to_csv(trade_log_path, mode='a', header=False, index=False)
            else:
                # Create new file with header
                df.to_csv(trade_log_path, index=False)
                
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")
            
    def _get_available_balance(self) -> float:
        """
        Get available USDT balance
        
        Returns:
            Available USDT balance
        """
        try:
            balances = self.account_info.get('balances', [])
            
            for balance in balances:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
                    
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting available balance: {e}")
            return 0.0
            
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        self.logger.info(f"Received signal {sig}. Initiating shutdown...")
        self.running = False
        
    def _shutdown(self):
        """Perform clean shutdown operations"""
        self.logger.info("Shutting down trading bot...")
        
        # Close any open positions
        if self.active_trade and self.current_position == Position.LONG:
            try:
                # Get latest data
                symbol = self.active_trade.symbol
                latest_price = self.binance_interface.get_ticker_price(symbol)
                
                # Create a mock DataFrame for the exit function
                data = pd.DataFrame({
                    'close': [latest_price],
                    'high': [latest_price],
                    'low': [latest_price],
                    'open': [latest_price],
                    'volume': [0]
                })
                
                # Execute exit
                self._execute_exit(None, data, "System shutdown")
            except Exception as e:
                self.logger.error(f"Error closing position during shutdown: {e}")
                
        # Generate final status report
        try:
            self._generate_status_report()
        except Exception as e:
            self.logger.error(f"Error generating final status report: {e}")
            
        # Close connections
        try:
            # Close Binance interface
            if self.binance_interface:
                self.binance_interface.close()
                
            # Wait for threads to finish (with timeout)
            threads = [
                self.data_thread,
                self.inference_thread,
                self.trading_thread,
                self.monitoring_thread
            ]
            
            for thread in threads:
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
                    
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
        self.logger.info("Trading bot shutdown complete.")


def load_config() -> Config:
    """
    Load configuration from command line arguments
    
    Returns:
        Configuration object
    """
    parser = argparse.ArgumentParser(description='Project Compound Trading Bot')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest'],
                      default='paper', help='Trading mode')
    parser.add_argument('--log-level', type=str, 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Logging level')
    parser.add_argument('--symbol', type=str, help='Override trading symbol')
    parser.add_argument('--timeframe', type=str, help='Override trading timeframe')
                      
    args = parser.parse_args()
    
    # Load config file
    config = Config(args.config)
    
    # Override with command line arguments
    config.set('bot.mode', args.mode)
    config.set('bot.log_level', args.log_level)
    
    if args.symbol:
        config.set('data.symbol', args.symbol)
        
    if args.timeframe:
        config.set('data.timeframe', args.timeframe)
        
    return config


def run_backtest(config: Config):
    """
    Run a backtest using historical data
    
    Args:
        config: Configuration object
    """
    logger = setup_logger("backtest", config.get('bot.log_level', 'INFO'))
    logger.info("Starting backtest mode...")
    
    # Load historical data
    data_path = config.get('backtest.data_path')
    if not data_path:
        logger.error("No data path specified for backtest. Exiting.")
        return
        
    try:
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.endswith('.pickle') or data_path.endswith('.pkl'):
            data = pd.read_pickle(data_path)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            return
            
        logger.info(f"Loaded historical data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
        
        # Initialize components
        feature_engineering = FeatureEngineering(config)
        trading_logic = TradingLogic(config)
        
        # Add features
        data = feature_engineering.add_all_features(data)
        
        # Initialize backtest variables
        initial_capital = config.get('backtest.initial_capital', 10000)
        current_capital = initial_capital
        equity_curve = [initial_capital]
        position = Position.FLAT
        active_trade = None
        trades = []
        
        # Run backtest
        logger.info(f"Running backtest with initial capital ${initial_capital:.2f}...")
        
        for i in range(100, len(data)):
            # Get current window
            current_data = data.iloc[:i+1]
            current_row = current_data.iloc[-1]
            current_price = current_row['close']
            current_time = current_data.index[-1]
            
            # Process with trading logic
            signal_result = trading_logic.update(
                current_data,
                current_capital,
                {Position.FLAT.name: 0, Position.LONG.name: 1}[position.name]
            )
            
            # Process signals
            if position == Position.FLAT and signal_result['action'] == 'ENTRY':
                # Enter position
                entry_price = current_price
                position_size = signal_result['position_size']
                stop_loss = signal_result['stop_loss']
                take_profit = signal_result['take_profit']
                
                # Create trade
                active_trade = Trade(
                    symbol=config.get('data.symbol', 'BTCUSDT'),
                    entry_price=entry_price,
                    entry_time=current_time,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                # Update state
                position = Position.LONG
                logger.info(
                    f"[{current_time}] ENTRY at ${entry_price:.2f}, "
                    f"Size: ${position_size:.2f}, "
                    f"Stop: ${stop_loss:.2f}, "
                    f"Target: ${take_profit:.2f}"
                )
                
            elif position == Position.LONG and (
                signal_result['action'] == 'EXIT' or
                current_price <= active_trade.stop_loss or
                current_price >= active_trade.take_profit
            ):
                # Determine exit reason
                if signal_result['action'] == 'EXIT':
                    exit_reason = signal_result['reason']
                elif current_price <= active_trade.stop_loss:
                    exit_reason = "Stop loss"
                elif current_price >= active_trade.take_profit:
                    exit_reason = "Take profit"
                else:
                    exit_reason = "Unknown"
                    
                # Close trade
                active_trade.close_trade(current_price, current_time, exit_reason)
                
                # Update capital
                pnl = active_trade.pnl
                current_capital += pnl
                
                # Log the exit
                logger.info(
                    f"[{current_time}] EXIT at ${current_price:.2f}, "
                    f"P/L: ${pnl:.2f} ({active_trade.pnl_percent:.2f}%), "
                    f"Reason: {exit_reason}"
                )
                
                # Add to trades list
                trades.append(active_trade)
                
                # Reset state
                position = Position.FLAT
                active_trade = None
                
            # Update equity curve
            equity_curve.append(current_capital)
            
        # Calculate performance metrics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(trades)
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            profit_factor = (
                sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades))
                if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
            )
            
            # Calculate max drawdown
            peak = initial_capital
            drawdowns = []
            
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                drawdowns.append(drawdown)
                
            max_drawdown = max(drawdowns)
            
            # Calculate Sharpe ratio
            equity_array = np.array(equity_curve)
            returns = np.diff(equity_array) / equity_array[:-1]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Print performance summary
            logger.info("\nBacktest Performance Summary:")
            logger.info(f"Initial Capital: ${initial_capital:.2f}")
            logger.info(f"Final Capital: ${current_capital:.2f}")
            logger.info(f"Total Return: {(current_capital / initial_capital - 1) * 100:.2f}%")
            logger.info(f"Total Trades: {len(trades)}")
            logger.info(f"Win Rate: {win_rate:.2f}")
            logger.info(f"Average Win: ${avg_win:.2f}")
            logger.info(f"Average Loss: ${avg_loss:.2f}")
            logger.info(f"Profit Factor: {profit_factor:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            
            # Save backtest results
            results_dir = config.get('backtest.results_dir', 'backtest_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # Save equity curve
            equity_df = pd.DataFrame({
                'equity': equity_curve,
                'drawdown': drawdowns
            })
            equity_df.to_csv(os.path.join(results_dir, f"equity-{timestamp}.csv"))
            
            # Save trades
            trades_data = [t.to_dict() for t in trades]
            with open(os.path.join(results_dir, f"trades-{timestamp}.json"), 'w') as f:
                json.dump(trades_data, f, indent=2)
                
            # Save performance metrics
            metrics = {
                'initial_capital': initial_capital,
                'final_capital': current_capital,
                'total_return': (current_capital / initial_capital - 1) * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'config': config.to_dict()
            }
            
            with open(os.path.join(results_dir, f"metrics-{timestamp}.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Backtest results saved to {results_dir}")
            
        else:
            logger.warning("No trades executed during backtest")
            
    except Exception as e:
        logger.error(f"Error in backtest: {e}")


def main():
    """Main entry point for the trading bot"""
    # Load configuration
    config = load_config()
    
    # Set up global logger
    global logger
    logger = setup_logger("main", config.get('bot.log_level', 'INFO'))
    
    # Check trading mode
    mode = config.get('bot.mode', 'paper')
    logger.info(f"Starting Project Compound Trading Bot in {mode} mode")
    
    if mode == 'backtest':
        # Run backtest
        run_backtest(config)
    else:
        # Run live or paper trading
        try:
            # Create and start trading bot
            bot = TradingBot(config.path, config.get('bot.log_level', 'INFO'))
            bot.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
            return 1
            
    return 0


if __name__ == "__main__":
    sys.exit(main())