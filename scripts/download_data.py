"""Download and extract CIFAR-10 dataset into the package data folder.

This script downloads the dataset from the canonical source and extracts the
`cifar-10-batches-py` folder into `pa2/data/`.
"""
import os
import urllib.request
import tarfile

def download_and_extract_cifar(dest_dir: str | None = None):
    if dest_dir is None:
        dest_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
    os.makedirs(dest_dir, exist_ok=True)

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tarfile_path = os.path.join(dest_dir, 'cifar-10-python.tar.gz')
    extract_path = os.path.join(dest_dir)

    if os.path.exists(os.path.join(dest_dir, 'cifar-10-batches-py')):
        print('CIFAR-10 already downloaded and extracted at', dest_dir)
        return

    print('Downloading CIFAR-10...')
    urllib.request.urlretrieve(url, tarfile_path)
    print('Download complete. Extracting...')
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print('Extract finished.')
    # remove the archive to save space
    try:
        os.remove(tarfile_path)
    except Exception:
        pass

if __name__ == '__main__':
    download_and_extract_cifar()
