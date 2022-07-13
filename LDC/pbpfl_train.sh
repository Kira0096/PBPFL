#training resnet20 model based on PBPFL on 1/5/10 clients, respectively

mkdir -p checkpoints/enc_res20_1; mkdir -p checkpoints/enc_res20_5; mkdir -p checkpoints/enc_res20_10;
python fl_train.py --enc -k 1 --arch resnet20 --seed 0 --save-dir checkpoints/enc_res20_1/ | tee checkpoints/enc_res20_1/log.txt
python fl_train.py --enc -k 5 --arch resnet20 --seed 0 --save-dir checkpoints/enc_res20_5/ | tee checkpoints/enc_res20_5/log.txt
python fl_train.py --enc -k 10 --arch resnet20 --seed 0 --save-dir checkpoints/enc_res20_10/ | tee checkpoints/enc_res20_10/log.txt