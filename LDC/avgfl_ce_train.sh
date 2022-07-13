#training resnet20 model based on AVGFL-CE on 1/5/10 clients, respectively

mkdir -p checkpoints/ce_res20_1; mkdir -p checkpoints/ce_res20_5; mkdir -p checkpoints/ce_res20_10;
python fl_train.py -k 1 --arch resnet20 --ce --seed 0 --save-dir checkpoints/ce_res20_1/ | tee checkpoints/ce_res20_1/log.txt
python fl_train.py -k 5 --arch resnet20 --ce --seed 0 --save-dir checkpoints/ce_res20_5/ | tee checkpoints/ce_res20_5/log.txt
python fl_train.py -k 10 --arch resnet20 --ce --seed 0 --save-dir checkpoints/ce_res20_10/ | tee checkpoints/ce_res20_10/log.txt