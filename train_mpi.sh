export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 python train.py --data-dir ../datasets/cityscape --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 0.01 --input-size 769,769 --weight-decay 0.0001 --batch-size 4 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000 --model ccnet >../log/train_resnet_b4.log
