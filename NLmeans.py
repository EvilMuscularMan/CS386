# encoding:utf-8
import process as pr

main_path = "/home/leslie/Documents/"

for i in range(1,5):
    src_path = main_path+"dataset/"+str(i)+"/"
    dst_path = main_path+"output/"+str(i)+"/"
    pr.mkdir(dst_path)
    for j in range(127,257):
        index = str((j//100))+str((j%100)//10)+str(j%10)
        if j<127 or j>129:
            pr.process(src_path, dst_path, index, 0, 0, 0)
        else:
            pr.process(src_path, dst_path, index, 1, 0, 1)
