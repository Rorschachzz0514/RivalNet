"""MPCC：为 DPPDCC 的额外随机种子建独立的数据源目录（全部数据与图用符号链接共享），避免不同种子的检查点 / 结果文件同名覆盖。
用法：python prep_seed_copy.py <源数据源> <新数据源>"""
import os, sys, shutil
src, dst = sys.argv[1], sys.argv[2]
for a, b in (("./data/%s/" % src, "./data/%s/" % dst), ("./checkpoints/%s/" % src, "./checkpoints/%s/" % dst)):
    os.makedirs(b, exist_ok=True)
    for f in os.listdir(a):
        if not f.endswith(".pkl") and not os.path.exists(b + f):
            os.symlink(os.path.realpath(a + f), b + f)
shutil.copy("./configs/%s.json" % src, "./configs/%s.json" % dst)
os.makedirs("./results/%s" % dst, exist_ok=True); os.makedirs("./logs/%s" % dst, exist_ok=True)
print("ok", dst)
