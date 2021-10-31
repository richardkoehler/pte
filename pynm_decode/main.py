"""Main module"""

import pynm_decode

if __name__ == "__main__":
    DIRECTORY = r"C:\Users\richa\OneDrive - Charité - Universitätsmedizin Berlin\PROJEC~3\PIPELI~1\DERIVA~1\FE94DF~1\m1_files"
    file_reader = pynm_decode.get_filereader(datatype="any")
    file_reader.find_files(directory=DIRECTORY)
    print("`pynm_decode` imported successfully.")
