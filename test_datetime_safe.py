#!/usr/bin/env python3
import sys
sys.path.append(r'D:\MCP_SERVER\INSTALLED\ollama-mcp-server\src')

from datetime import datetime
import json

# Test della funzione current _format_datetime_safe
def _format_datetime_safe(dt_value):
    """Convert datetime objects to string format safely, using simple str() conversion."""
    if isinstance(dt_value, datetime):
        return str(dt_value)
    elif isinstance(dt_value, dict):
        # Recursively clean datetime objects in dictionaries
        clean_dict = {}
        for k, v in dt_value.items():
            clean_dict[k] = _format_datetime_safe(v)
        return clean_dict
    elif isinstance(dt_value, list):
        # Recursively clean datetime objects in lists
        return [_format_datetime_safe(item) for item in dt_value]
    else:
        return dt_value

# Test con un datetime object
dt = datetime(2025, 8, 6, 16, 13, 0, 921606)
print(f"Datetime object: {dt} (type: {type(dt)})")

converted = _format_datetime_safe(dt)
print(f"Converted: {converted} (type: {type(converted)})")

# Test JSON serialization
try:
    json_str = json.dumps(converted)
    print(f"JSON serialization: SUCCESS")
    print(f"JSON: {json_str}")
except Exception as e:
    print(f"JSON serialization: FAILED - {e}")