export CUDA_VISIBLE_DEVICES=4

# Resnet + CCNet + Cityscape
# python train.py --data-dir ../datasets/cityscape --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 4 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 1 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet >../log/train_resnet_CS_b1.log

# Resnet + CCNet + ADE20K
python train.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --random-mirror --restore-from ./dataset/resnet101-imagenet.pth --data-list ./dataset/list/ade20k/ade20k_train.txt --gpu 4 --learning-rate 0.01 --input-size 600,600 --weight-decay 0.0001 --batch-size 1 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet --need_crop >../log/train_resnet_ADE_b1.log

# VAN + CCNet + Cityscape
# python train.py --data-dir ../datasets/cityscape --random-mirror --random-scale --restore-from ./dataset/van_b1.pth --gpu 4 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 1 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model van > ../log/train_van_CS_b1.log

# VAN + CCNet + ADE20K
# python train.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --random-mirror --random-scale --restore-from ./dataset/van_b1.pth --data-list ./dataset/list/ade20k/ade20k_train.txt --gpu 4 --learning-rate 0.01 --input-size 600,600 --weight-decay 0.0001 --batch-size 1 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model van --need_crop > ../log/train_van_ADE_b1.log