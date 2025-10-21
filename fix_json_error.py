# Fix for JSON Serialization Error
# Add this helper function to your notebook

import numpy as np
import json

def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Then when saving results, replace:
# json.dump(results, f, indent=2)

# With:
json.dump(convert_to_serializable(results), f, indent=2)

# OR use this simpler one-liner:
# json.dump(results, f, indent=2, default=float)  # Converts all numeric types to float
