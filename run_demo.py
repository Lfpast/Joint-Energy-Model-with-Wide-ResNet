"""
Quick demo runner to check structure and import for the `pa2` package.
"""
import os
from pa2.data.preprocess import load_cifar10

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'cifar-10-batches-py')
    if not os.path.exists(data_dir):
        # Try to download
        from pa2.scripts.download_data import download_and_extract_cifar
        download_and_extract_cifar(os.path.join(base_dir, 'data'))

    print('PA2 package demo:')
    try:
        # test imports
        import pa2 as _
        print('pa2 imported successfully')
        X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
        print('Data loaded: ', X_train.shape, Y_train.shape)
    except Exception as e:
        print('Import error:', e)

if __name__ == '__main__':
    main()
