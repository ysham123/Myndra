"""Quick pipeline test for radiology models."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domains.radiology_pneumonia.pipeline import predict as predict_pneumonia
from domains.radiology_cardiomegaly.pipeline import predict as predict_cardiomegaly

def main():
    """Test both radiology pipelines with a sample image."""
    # Use a test image if available
    test_images = [
        "tests/assets/sample_cxr.jpg",
        "image.png",  # fallback to root image if test image not found
    ]
    
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found. Please add a test image.")
        return 1
    
    print(f"Testing with image: {test_image}\n")
    
    # Test pneumonia pipeline
    print("🫁 Testing Pneumonia Detection Pipeline...")
    try:
        result = predict_pneumonia(test_image)
        print(f"  ✅ Diagnosis: {result['diagnosis']}")
        print(f"  ✅ Probability: {result['probability']:.2%}")
        print(f"  ✅ Steps: {len(result['steps'])} processing steps")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 1
    
    print()
    
    # Test cardiomegaly pipeline
    print("❤️  Testing Cardiomegaly Detection Pipeline...")
    try:
        result = predict_cardiomegaly(test_image)
        print(f"  ✅ Diagnosis: {result['diagnosis']}")
        print(f"  ✅ Probability: {result['probability']:.2%}")
        print(f"  ✅ Steps: {len(result['steps'])} processing steps")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 1
    
    print("\n✅ All pipeline tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
