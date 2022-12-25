import time
import jittor as jt
# from networks.ccnet import Seg_Model
from networks.van import Seg_Model
from utils.pyt_utils import load_model

jt.flags.use_cuda = jt.has_cuda

# 模型参数
num_classes = 19
recurrence = 2
restore_from = './snapshots/CS_scenes_15000_van.pkl'

warmup = 100
rerun = 1000
batch_size = 1
data = jt.random((batch_size, 3, 769, 769))
model = Seg_Model(
            num_classes=num_classes, recurrence=recurrence
        )
load_model(model, restore_from)
model.eval()

# 此段代码对jittor进行热身，确保时间测试准确
jt.sync_all(True)
for i in range(warmup):
    pred = model(data)[0]
    # sync是把计算图发送到计算设备上
    pred.sync()
# sync_all(true)是把计算图发射到计算设备上，并且同步。
# 只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
jt.sync_all(True)

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = model(data)[0]
    pred.sync()
jt.sync_all(True)
end = time.time()

print("Jittor FPS:", (rerun*batch_size)/(end-start))