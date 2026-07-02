# 下载远程服务器文件到本地的命令:
rsync -avzP --partial \
  <远程别名>:<远程路径> \
  <本地路径>

# 上传本地文件到远程服务器的命令:
rsync -avzP --partial \
  <本地路径> \
  <远程别名>:<远程路径>

# 传整个文件夹
rsync -avzP --partial my_folder/ 4090:/target/path/

# 只传某个文件
rsync -avzP --partial my_file.ckpt 4090:/target/path/

# 排除某些文件（如缓存、git）
rsync -avzP --partial --exclude='__pycache__' --exclude='.git' project/ 4090:/target/project/

# 同步（删除远程多余文件，慎用）
rsync -avzP --partial --delete project/ 4090:/target/project/

# 查看log
tensorboard --logdir logs/sd2_control_ldm/

python -c "
import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

log_dir = 'logs/sd2_control_ldm'
versions = sorted(glob.glob(f'{log_dir}/version_*'))
print('可用版本:', versions)
target = versions[-1] if versions else None
if not target:
    print('没有找到任何 TensorBoard 日志')
    exit(1)

print(f'读取: {target}')
ea = EventAccumulator(target)
ea.Reload()

rows = {}
for tag in ea.Tags()['scalars']:
    for e in ea.Scalars(tag):
        rows.setdefault(e.step, {})[tag] = e.value

result = pd.DataFrame.from_dict(rows, orient='index').sort_index()
result.index.name = 'step'
print(f'共 {len(result)} 行, {len(result.columns)} 个指标')
print()
print(result.tail(15).to_string())
result.to_csv('training_log_export.csv')
print(f'\n已保存 training_log_export.csv')
"
