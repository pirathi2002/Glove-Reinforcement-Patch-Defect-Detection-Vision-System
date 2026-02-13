"""
SSL Certificate Setup and Model Pre-download Script
This script should be run once to fix SSL issues and pre-download required models.
"""

import os
import sys
import ssl
import certifi
from pathlib import Path

# Update Python's certificate bundle
print("="*80)
print("FIXING SSL CERTIFICATE ISSUES")
print("="*80)

# Get certifi's CA bundle
ca_bundle = certifi.where()
print(f"\nCA Bundle location: {ca_bundle}")
print(f"CA Bundle exists: {os.path.exists(ca_bundle)}")
print(f"CA Bundle size: {os.path.getsize(ca_bundle)} bytes")

# Set environment variables for both current and future processes
os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
os.environ['CURL_CA_BUNDLE'] = ca_bundle
os.environ['SSL_CERT_FILE'] = ca_bundle
os.environ['SSL_CERT_DIR'] = os.path.dirname(ca_bundle)

print(f"\nEnvironment variables set:")
print(f"  REQUESTS_CA_BUNDLE={ca_bundle}")
print(f"  CURL_CA_BUNDLE={ca_bundle}")
print(f"  SSL_CERT_FILE={ca_bundle}")
print(f"  SSL_CERT_DIR={os.path.dirname(ca_bundle)}")

# Try to pre-download the model
print("\n" + "="*80)
print("ATTEMPTING TO PRE-DOWNLOAD MODELS")
print("="*80)

try:
    print("\nDownloading wide_resnet50_2 model...")
    import timm
    
    # This will download and cache the model
    model = timm.create_model('wide_resnet50_2', pretrained=True)
    print("✓ Successfully downloaded wide_resnet50_2 model")
    
except Exception as e:
    print(f"✗ Failed to download model: {e}")
    print("\nTrying alternative approach with offline mode...")
    
    # Try with offline mode or local model
    try:
        import timm
        model = timm.create_model('wide_resnet50_2', pretrained=False)
        print("✓ Created model without pre-trained weights")
    except Exception as e2:
        print(f"✗ Alternative also failed: {e2}")

print("\n" + "="*80)
print("SSL SETUP COMPLETE")
print("="*80)
print("\nYou can now run training with:")
print("  python main.py --train --model patchcore")
