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
tensorboard --logdir logs/sd2_control
