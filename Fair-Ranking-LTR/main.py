import pyterrier as pt
from data import *
import os

def main():
    print('Heello')


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    
    pt.init()
    print('Pyterrier started')
    downloaded_data = False
    if downloaded_data == False:
        
        #download_data_p()
        download_data('irds:trec-fair/2022')
        
    main()


