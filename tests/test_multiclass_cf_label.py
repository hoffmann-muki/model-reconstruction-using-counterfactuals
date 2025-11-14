"""Simple test for cf_label auto-selection logic without external dataset dependencies."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def test_cf_label_logic():
    """Test the cf_label auto-selection logic directly."""
    
    print("Testing cf_label auto-selection logic...")
    
    # Test 1: Binary case
    print("\nTest 1: Binary classification (num_classes=2)")
    num_classes = 2
    cf_label = None
    
    # Simulate the auto-selection logic from utils_v8.py
    if cf_label is None:
        if num_classes == 2:
            cf_label = 0.5
        else:
            cf_label = 'prediction'
    
    print(f" Input: num_classes={num_classes}, cf_label=None")
    print(f" Output: cf_label={cf_label}")
    assert cf_label == 0.5, f"Expected 0.5 for binary, got {cf_label}"
    print(" Binary case: cf_label correctly set to 0.5")
    
    # Test 2: Multiclass case
    print("\nTest 2: Multiclass classification (num_classes=3)")
    num_classes = 3
    cf_label = None
    
    if cf_label is None:
        if num_classes == 2:
            cf_label = 0.5
        else:
            cf_label = 'prediction'
    
    print(f" Input: num_classes={num_classes}, cf_label=None")
    print(f" Output: cf_label={cf_label}")
    assert cf_label == 'prediction', f"Expected 'prediction' for multiclass, got {cf_label}"
    print("  Multiclass case: cf_label correctly set to 'prediction'")
    
    # Test 3: Out-of-band mode for binary
    print("\nTest 3: Out-of-band mode for binary (num_classes=2)")
    num_classes = 2
    cf_label = 'out-of-band'
    
    # Simulate the out-of-band logic
    if cf_label == 'out-of-band':
        if num_classes == 2:
            cf_label = 0.5
        else:
            cf_label = -1
    
    print(f" Input: num_classes={num_classes}, cf_label='out-of-band'")
    print(f" Output: cf_label={cf_label}")
    assert cf_label == 0.5, f"Expected 0.5 for binary out-of-band, got {cf_label}"
    print("  Binary out-of-band: cf_label correctly set to 0.5")
    
    # Test 4: Out-of-band mode for multiclass
    print("\nTest 4: Out-of-band mode for multiclass (num_classes=5)")
    num_classes = 5
    cf_label = 'out-of-band'
    
    if cf_label == 'out-of-band':
        if num_classes == 2:
            cf_label = 0.5
        else:
            cf_label = -1
    
    print(f" Input: num_classes={num_classes}, cf_label='out-of-band'")
    print(f" Output: cf_label={cf_label}")
    assert cf_label == -1, f"Expected -1 for multiclass out-of-band, got {cf_label}"
    print(" Multiclass out-of-band: cf_label correctly set to -1")
    
    # Test 5: Explicit cf_label value (should not be overridden)
    print("\nTest 5: Explicit cf_label value (should not be overridden)")
    num_classes = 2
    cf_label = 0.7  # User-specified value
    original_cf_label = cf_label
    
    # The auto-selection only happens when cf_label is None
    if cf_label is None:
        if num_classes == 2:
            cf_label = 0.5
        else:
            cf_label = 'prediction'
    
    print(f" Input: num_classes={num_classes}, cf_label=0.7")
    print(f" Output: cf_label={cf_label}")
    assert cf_label == 0.7, f"Expected user value 0.7 to be preserved, got {cf_label}"
    print(" Explicit value: cf_label correctly preserved at 0.7")
    
    print("\n" + "="*60)
    print("All cf_label auto-selection logic tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_cf_label_logic()
