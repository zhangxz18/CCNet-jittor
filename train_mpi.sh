export CUDA_VISIBLE_DEVICES=0,1,2,3

# Resnet + CCNet + Cityscape
# mpirun -np 4 python train.py --data-dir ../datasets/cityscape --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 4 --num-steps 15000 --save-pred-every 5000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet >../log/train_resnet_CS_b4.log

# Resnet + CCNet + ADE20K
mpirun -np 4 python train.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --random-mirror --restore-from ./dataset/resnet101-imagenet.pth --data-list ./dataset/list/ade20k/ade20k_train.txt --gpu 0,1,2,3 --learning-rate 0.01 --input-size 600,600 --weight-decay 0.0001 --batch-size 4 --num-steps 15000 --save-pred-every 5000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet --need_crop >../log/train_resnet_ADE_b4.log

# VAN + CCNet + Cityscape
# mpirun -np 4 python train.py --data-dir ../datasets/cityscape --random-mirror --random-scale --restore-from ./dataset/van_b1.pth --gpu 0,1,2,3 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 4 --num-steps 15000 --save-pred-every 5000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model van > ../log/train_van_CS_b4.log

# VAN + CCNet + ADE20K
# mpirun -np 4 python train.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --random-mirror --random-scale --restore-from ./dataset/van_b1.pth --data-list ./dataset/list/ade20k/ade20k_train.txt --gpu 0,1,2,3 --learning-rate 0.01 --input-size 600,600 --weight-decay 0.0001 --batch-size 4 --num-steps 15000 --save-pred-every 5000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model van --need_crop > ../log/train_van_ADE_b4.log