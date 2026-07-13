from PIL import Image as _Image

# HiReS reads trusted local scans far larger than PIL's ~89 MP bomb guard.
_Image.MAX_IMAGE_PIXELS = None
