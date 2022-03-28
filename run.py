import warnings
from extract import extract
from compress import compress
from reconstruct import reconstruct
from reid import reid
from main import xxx

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    extract()
    br = [64, 128, 256]
    for byte in br:
        print('\n- - - - - ' + str(byte) + ' - - - - -')
        compress(byte)
        reconstruct(byte)
        xxx(byte)
        reid(byte)
