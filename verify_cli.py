import sys
from verify_pair import *


if len(sys.argv) != 3:
    print("Usage: python verify_cli.py img1.jpg img2.jpg")
    exit()

img1 = sys.argv[1]
img2 = sys.argv[2]

# langsung gunakan verify_pair
