"""Download CAMELS-US (~1GB zip) to data/raw/. Verify the URL for your mirror."""
from pathlib import Path
import urllib.request, zipfile

URL = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels.tar.gz"
OUT = Path("data/raw/camels.tar.gz")
OUT.parent.mkdir(parents=True, exist_ok=True)
if not OUT.exists():
    print("downloading (may take a few minutes on Colab)...")
    urllib.request.urlretrieve(URL, OUT)
print("extracting...")
import tarfile
with tarfile.open(OUT) as t:
    t.extractall("data/raw/")
print("done ->", list(Path("data/raw").iterdir()))
