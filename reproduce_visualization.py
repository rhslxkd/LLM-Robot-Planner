import sys
import os
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

# Mock the config module BEFORE importing anything from vlm_courtroom that uses it
sys.modules['vlm_courtroom.config'] = MagicMock()
sys.modules['vlm_courtroom.config'].get_model = MagicMock(return_value=MagicMock())

from vlm_courtroom.court.courtroom import VLMCourt

def test_visualization():
    print("Initializing VLMCourt for testing...")
    court = VLMCourt()
    
    # Mock image path - using a dummy file
    dummy_image_path = "dummy_test_image.png"
    plt.imsave(dummy_image_path, np.zeros((100, 100, 3)))
    
    print(f"Created dummy image at {dummy_image_path}")

    # Test Case 1: JSON with single quotes (Common LLM error)
    verdict_1 = """
    Here is the verdict.
    coordinates: [{'x': 10, 'y': 10}, {'x': 20, 'y': 20}]
    """
    
    print("\n--- Test 1: Single Quotes ---")
    try:
        court.visualize_path(dummy_image_path, verdict_1)
        print("✅ Test 1 Passed (Single Quotes handled)")
    except Exception as e:
        print(f"❌ Test 1 Failed: {e}")
    
    # Test Case 2: Markdown block
    verdict_2 = """
    The path is:
    ```json
    [{"x": 30, "y": 30}, {"x": 40, "y": 40}]
    ```
    """
    print("\n--- Test 2: Markdown Block ---")
    try:
        court.visualize_path(dummy_image_path, verdict_2)
        print("✅ Test 2 Passed (Markdown Block handled)")
    except Exception as e:
        print(f"❌ Test 2 Failed: {e}")

    # Test Case 3: Both (Markdown + Single Quotes)
    verdict_3 = """
    ```json
    [{'x': 50, 'y': 50}]
    ```
    """
    print("\n--- Test 3: Markdown + Single Quotes ---")
    try:
        court.visualize_path(dummy_image_path, verdict_3)
        print("✅ Test 3 Passed (Markdown + Single Quotes handled)")
    except Exception as e:
        print(f"❌ Test 3 Failed: {e}")

    # Cleanup
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
    
    print("\nAll Tests Complete.")

if __name__ == "__main__":
    test_visualization()
