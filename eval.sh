export CUDA_VISIBLE_DEVICES=0,1,2,3

# Resnet + CCNet + Cityscape
# python evaluate.py --data-dir ../datasets/cityscape  --recurrence 2 --model ccnet --restore-from ./snapshots/CS_scenes_40000ccnet.pkl --whole True --gpu 2 --batch-size 1 >../log/eval_resnet_CS_4w.log

# Resnet + CCNet + ADE20K
python evaluate.py --num-classes 150 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --data-list ./dataset/list/ade20k/ade20k_val.txt --recurrence 2 --model ccnet --restore-from ./snapshots/ADE20k_80000_ccnet.pkl --gpu 0 --batch-size 1 >../log/eval_resnet_ADE_b8_8w_retest.log

# VAN + CCNet + Cityscape
# python evaluate.py --data-dir ../datasets/cityscape  --recurrence 2 --model van --restore-from ./snapshots/CS_scenes_15000van.pkl --whole True --gpu 0 --batch-size 1 >../log/eval_van_CS.txt

# VAN + CCNet + ADE20K
# python evaluate.py --num-classes 151 --datasets ade --data-dir ../datasets/ADEChallengeData2016 --data-list ./dataset/list/ade20k/ade20k_val.txt --recurrence 2 --model van --restore-from ./snapshots/ADE20k_15000van.pkl --gpu 0 --batch-size 1 >../log/eval_van_ADE.log