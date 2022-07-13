#training MLP model based on AVGFL on 1/5/10 clients, respectively

mkdir -p checkpoints/avg_mlp3_1; mkdir -p checkpoints/avg_mlp3_5; mkdir -p checkpoints/avg_mlp3_10

python fl_train.py -k 1 --arch 3 -b 512   --lr 1e-3 --enc  --save-dir checkpoints/avg_mlp3_1/ | tee checkpoints/avg_mlp3_1/log.txt
python fl_train.py -k 5 --arch 3 -b 512   --lr 1e-3 --enc  --save-dir checkpoints/avg_mlp3_5/ | tee checkpoints/avg_mlp3_5/log.txt
python fl_train.py -k 10 --arch 3 -b 512  --lr 1e-3  --enc  --save-dir checkpoints/avg_mlp3_10/ | tee checkpoints/avg_mlp3_10/log.txt

