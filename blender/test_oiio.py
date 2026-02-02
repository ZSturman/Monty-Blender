import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nTrying to import OpenImageIO...")
try:
    import OpenImageIO as oiio
    print(f"✓ OpenImageIO found: {oiio}")
    print(f"  Version: {oiio.VERSION if hasattr(oiio, 'VERSION') else 'unknown'}")
except ImportError as e:
    print(f"✗ OpenImageIO not available: {e}")
