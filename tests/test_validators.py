import pytest
from src.data.tick_validator import TickValidator

def test_validator_valid_tick():
    """Test validation of a valid tick."""
    validator = TickValidator()

    tick = {
        "instrument_token": 12345,
        "last_price": 100.5,
        "volume": 1000,
        "ohlc": {
            "open": 99.0,
            "high": 101.0,
            "low": 98.5,
            "close": 100.5
        }
    }

    is_valid, error = validator.validate_tick(tick)
    assert is_valid
    assert error is None

def test_validator_missing_field():
    """Test validation with missing field."""
    validator = TickValidator()

    tick = {
        "instrument_token": 12345,
        "last_price": 100.5,
        # Missing 'volume'
    }

    is_valid, error = validator.validate_tick(tick)
    assert not is_valid
    assert error is not None

def test_validator_invalid_price():
    """Test validation with invalid price."""
    validator = TickValidator()

    tick = {
        "instrument_token": 12345,
        "last_price": -100.5,  # Negative price
        "volume": 1000,
        "ohlc": {}
    }

    is_valid, error = validator.validate_tick(tick)
    assert not is_valid
    assert "positive" in error.lower()

def test_validator_outlier_detection():
    """Test outlier detection."""
    validator = TickValidator()

    # First tick
    tick1 = {
        "instrument_token": 12345,
        "last_price": 100.0,
        "volume": 1000,
        "ohlc": {}
    }
    is_valid1, _ = validator.validate_tick(tick1)
    assert is_valid1

    # Second tick with >5% jump (should be flagged as outlier)
    tick2 = {
        "instrument_token": 12345,
        "last_price": 106.0,  # 6% increase
        "volume": 1000,
        "ohlc": {}
    }
    is_valid2, error = validator.validate_tick(tick2)
    assert not is_valid2
    assert "outlier" in error.lower()

def test_validator_batch():
    """Test batch validation."""
    validator = TickValidator()

    ticks = [
        {
            "instrument_token": 12345,
            "last_price": 100.0,
            "volume": 1000,
            "ohlc": {}
        },
        {
            "instrument_token": 12345,
            "last_price": 101.0,
            "volume": 1100,
            "ohlc": {}
        },
        {
            "instrument_token": 12345,
            "last_price": -50.0,  # Invalid
            "volume": 900,
            "ohlc": {}
        }
    ]

    valid, invalid = validator.validate_batch(ticks)
    assert len(valid) == 2
    assert len(invalid) == 1
