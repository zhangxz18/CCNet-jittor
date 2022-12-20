export CUDA_VISIBLE_DEVICES=0

# Resnet + CCNet + Cityscape
# python evaluate.py --data-dir ../datasets/cityscape  --recurrence 2 --model ccnet --restore-from ./snapshots/CS_scenes_15000ccnet.pkl --whole True --gpu 0 --batch-size 1 >../log/eval_resnet_CS.log

# Resnet + CCNet + ADE20K
python evaluate.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --data-list ./dataset/list/ade20k/ade20k_val.txt --recurrence 2 --model ccnet --restore-from ./snapshots/ADE20k_15000ccnet.pkl --gpu 0 --batch-size 1 >../log/eval_resnet_ADE_size512.log

# VAN + CCNet + Cityscape
# python evaluate.py --data-dir ../datasets/cityscape  --recurrence 2 --model van --restore-from ./snapshots/CS_scenes_15000van.pkl --whole True --gpu 0 --batch-size 1 >../log/eval_van_CS_dsn.txt

# VAN + CCNet + ADE20K
# python evaluate.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --data-list ./dataset/list/ade20k/ade20k_val.txt --recurrence 2 --model van --restore-from ./snapshots/ADE20k_15000van.pkl --gpu 0 --batch-size 1 >../log/eval_van_ADE.log