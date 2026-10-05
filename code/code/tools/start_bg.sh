#!/bin/bash
# 用法: start_bg.sh <名称> <命令...>   后台启动并把进程组 ID 写入 logs/<名称>.pid; 停止用: kill -- -$(cat logs/<名称>.pid)
name=$1; shift
cd /path/to/mpcc/code
setsid nohup "$@" > /path/to/mpcc/logs/$name.log 2>&1 < /dev/null &
echo $! > /path/to/mpcc/logs/$name.pid
echo "started $name pid=$(cat /path/to/mpcc/logs/$name.pid)"
