import sys
import os
import json
import copy
import subprocess

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

with open('sparse_conv2.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))

#for i in range(100,128):
#for i in [6,10,16,18,26,28]: # addmm & mm
#for i in [2]: # sum
for i in range(len(route)): # embedding_back
#for i in [10,16,18]: # addmm & mm -- big
  cmd = copy.deepcopy(route)
  cmd[i]['offload'] = True
  fancy_print(cmd)
  print()
  name = "sparse_conv2_hb_%d" % i
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  with open(name + "/sparse_conv2.json", 'w') as outfile:
    json.dump(cmd, outfile, indent=4, sort_keys=True)
  sh_cmd = "cp LeNet_5.prune.conv.fc.pth.tar " + name + "/"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim -m pytest -vs /scratch/users/zz546/pytorch-cosim/hb-pytorch/hammerblade/torch/tests/profiler/test_lenet5_sparseconv2_profile_route.py > out.std 2>&1)"
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  for runs in range(3):
      cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)
      cosim_run.wait()
