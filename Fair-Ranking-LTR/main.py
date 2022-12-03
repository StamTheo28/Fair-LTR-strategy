import pyterrier as pt
from data import *

def main():
    print('Heello')


if __name__ == "__main__":
    if not pt.started:
        pt.init()
    download_data = False
    if download_data == False:
        download_data('irds:trec-fair-2021')
    main()


