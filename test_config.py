"""
Test script for config.py verification.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from . import config

def test_config_values():
    """Test if config values are set correctly"""
    # Check trading settings
    assert config.SYMBOL == "BTC/USDC", f"Symbol is {config.SYMBOL}, expected BTC/USDC"
    assert config.QUOTE_CURRENCY == "USDC", f"Quote currency is {config.QUOTE_CURRENCY}, expected USDC"
    assert config.INITIAL_CAPITAL == 100.0, f"Initial capital is {config.INITIAL_CAPITAL}, expected 100.0"
    assert config.MAX_POSITION_SIZE == 1.0, f"Max position size is {config.MAX_POSITION_SIZE}, expected 1.0"
    assert config.STOP_LOSS_PCT == 0.004, f"Stop loss is {config.STOP_LOSS_PCT}, expected 0.004"
    assert config.TAKE_PROFIT_PCT == 0.04, f"Take profit is {config.TAKE_PROFIT_PCT}, expected 0.04"
    assert config.MAX_HOLDING_PERIOD == 24, f"Max holding period is {config.MAX_HOLDING_PERIOD}, expected 24"
    
    # Check inference settings
    assert config.INFERENCE_SETTINGS["max_positions"] == 1, f"Max positions is {config.INFERENCE_SETTINGS['max_positions']}, expected 1"
    assert "exit_threshold" in config.INFERENCE_SETTINGS, "Missing exit_threshold in inference settings"
    assert "peak_detection_window" in config.INFERENCE_SETTINGS, "Missing peak_detection_window in inference settings"
    
    # Check model settings
    assert "peak_prediction" in config.MODEL_SETTINGS, "Missing peak_prediction in model settings"
    assert "use_exit_model" in config.MODEL_SETTINGS, "Missing use_exit_model in model settings"
    assert "short_term_horizon" in config.MODEL_SETTINGS, "Missing short_term_horizon in model settings"
    
    print("All config values are set correctly!")

def test_config_functions():
    """Test config utility functions"""
    # Test get_config()
    config_dict = config.get_config()
    assert isinstance(config_dict, dict), "get_config() should return a dictionary"
    assert "SYMBOL" in config_dict, "SYMBOL should be in config dictionary"
    assert "MODEL_SETTINGS" in config_dict, "MODEL_SETTINGS should be in config dictionary"
    
    # Test update_config()
    original_timeframe = config.TIMEFRAME
    try:
        config.update_config({"TIMEFRAME": "5m"})
        assert config.TIMEFRAME == "5m", f"Failed to update TIMEFRAME, got {config.TIMEFRAME}, expected 5m"
    finally:
        # Restore original value
        config.update_config({"TIMEFRAME": original_timeframe})
    
    print("All config functions are working correctly!")

if __name__ == "__main__":
    print("Testing config.py...")
    test_config_values()
    test_config_functions()
    
    # Print current configuration
    print("\nCurrent Trading Configuration:")
    trading_config = {
        "SYMBOL": config.SYMBOL,
        "INITIAL_CAPITAL": config.INITIAL_CAPITAL,
        "MAX_POSITION_SIZE": config.MAX_POSITION_SIZE,
        "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
        "TAKE_PROFIT_PCT": config.TAKE_PROFIT_PCT,
        "TARGET_PROFIT_PCT": config.TARGET_PROFIT_PCT,
        "MAX_HOLDING_PERIOD": config.MAX_HOLDING_PERIOD
    }
    print(json.dumps(trading_config, indent=4))
    
    print("\nConfiguration test completed successfully!")