#!/bin/sh
python extract.py
echo '- - - - - - - - - -'
python compress.py
echo '- - - - - - - - - -'
python reconstruct.py
echo '- - - - - - - - - -'
python main.py
echo '- - - - - - - - - -'
python train_update.py
echo '- - - - - - - - - -'
python reid.py
echo '- - - - - - - - - -'