#training resnet20 model based on AVGFL-MSE on 1/5/10 clients, respectively

mkdir -p checkpoints/avg_res20_1; mkdir -p checkpoints/avg_res20_5; mkdir -p checkpoints/avg_res20_10;
python fl_train.py -k 1 --arch resnet20 --seed 0 --save-dir checkpoints/avg_res20_1/ | tee checkpoints/avg_res20_1/log.txt
python fl_train.py -k 5 --arch resnet20 --seed 0 --save-dir checkpoints/avg_res20_5/ | tee checkpoints/avg_res20_5/log.txt
python fl_train.py -k 10 --arch resnet20 --seed 0 --save-dir checkpoints/avg_res20_10/ | tee checkpoints/avg_res20_10/log.txt