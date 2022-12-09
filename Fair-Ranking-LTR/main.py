import pyterrier as pt
from data import *
import os

def main():
    print('Heello')


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-19"
    if not pt.started:
        pt.init()
    print('Pyterrier started')
    downloaded_data = False
    if downloaded_data == False:
        download_data('irds:trec-fair/2021')
    main()


