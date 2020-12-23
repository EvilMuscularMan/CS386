import cv2 as cv
import math
import os
import numpy as np
from tqdm import tqdm

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---["+path+"]created!---")

def sharpen(Img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    ans = cv.filter2D(Img, -1, kernel=kernel)
    return ans

#to calculate the distance between two imgs
def distance(img1, img2):
    r = img1.shape[0]
    c = img1.shape[1]
    if r!=img2.shape[0] or c!=img2.shape[1]:
        print("error:different size!")
        print("     %d*%d  %d*%d" %(r, c, img2.shape[0], img2.shape[1]))
    ans = 0
    for x in range(0, r):
        for y in range(0, c):
            delta = int(img1[x, y])-int(img2[x, y])
            ans += delta * delta
    return ans

#update one pixel according to its search area(partImg)
#h is coefficient of goss function
#d is the radius of neighborhood(a square of 2*d+1)
def NL_update(h, d, partImg):
    l = partImg.shape[0]
    D = l // 2
    distSum = 0
    weightSum = 0
    gossmax = 0
    img1 = partImg[D-d:D+d+1, D-d:D+d+1]
    for x in range(d, l-d):
        for y in range(d, l-d):
            if x==D and y==D:
                continue
            img2 = partImg[x-d:x+d+1, y-d:y+d+1]
            res = distance(img1, img2)
            goss = math.exp(-1 * res / (h * h))
            if goss>gossmax:
                gossmax = goss
            weightSum += goss * partImg[x, y]
            distSum += goss
    weightSum += gossmax * partImg[D, D]
    distSum += gossmax
    return weightSum/distSum

def NL_means(img, D, d, h):
    r, c = img.shape[0:2]
    Img = cv.copyMakeBorder(img, D, D, D, D, cv.BORDER_CONSTANT, 0)
    newImg = img.copy()
    for row in tqdm(range(0, r)):
        for col in range(0, c):
            partImg = Img[row:row+2*D+1, col:col+2*D+1]
            newImg[row, col] = NL_update(h, d, partImg)
    
    clahe = cv.createCLAHE(3,(8,8))
    #newimg = clahe.apply(newImg)
    return newImg


def get_src(src_path, index):
    return src_path+"bscan_"+index+".jpg"

def get_dst(dst_path, filter_type, index):
    return dst_path+filter_type+"_"+index+".jpg"

def get_Bscan(src_path):
    img = cv.imread(src_path+"B-sacn.jpg", cv.IMREAD_GRAYSCALE)
    r,c = img.shape[0:2]
    M = np.float32([[0.5, 0, 0], [0, 1, 0]])
    normalize_img = cv.warpAffine(img, M, (c//2, r))
    return normalize_img

def get_PSNR(OriginalImg, NoisyImg):
    r, c = OriginalImg.shape[0:2]
    MSE = distance(OriginalImg, NoisyImg)/(r*c)
    return 10*math.log10(255*255/MSE)

#this function is used to get an img at src, then process it according to the #filter_type, and store it at dst_path, named with its index
#show_flag = true: there's a window for each img
#filter_type : 0 = all, 1 = NL, 2 = median, 3 = bilateral
#PSNR_flag = true: calculate PSNR
def process(src_path, dst_path, index, show_flag, filter_type, PSNR_flag):
    src = get_src(src_path, index)
    print("open:"+src)
    img = cv.imread(src, cv.IMREAD_GRAYSCALE)
    NNimg =  cv.imread(src_path+"Self2Self-43000.png", cv.IMREAD_GRAYSCALE)
    if PSNR_flag:
        Bscan = get_Bscan(src_path) 
        ans = get_PSNR(Bscan, img)
        print("PSNR of originalImg:"+str(ans))
        ans = get_PSNR(Bscan, NNimg)
        print("PSNR of NNImg:"+str(ans))
    if show_flag:
        cv.imshow("img",img)  
        cv.imshow("normalImg", Bscan)     
    
    if filter_type==0 or filter_type==1:
        NLImg = NL_means(img, 5, 2, 100)
        #NLImg = sharpen(newImg)
        if show_flag:
            cv.imshow("NLImg",NLImg)
        if PSNR_flag:
            ans = get_PSNR(Bscan, NLImg)
            print("PSNR of NL-means:"+str(ans))
        dst = get_dst(dst_path, "NL", index)
        cv.imwrite(dst, NLImg)
        print("save to:"+dst)
    
    if filter_type==0 or filter_type==2:
        medianImg = cv.medianBlur(img, 5)
        if show_flag:
            cv.imshow("mediaImg",medianImg)
        if PSNR_flag:
            ans = get_PSNR(Bscan, medianImg)
            print("PSNR of median:"+str(ans))
        dst = get_dst(dst_path, "median", index)
        cv.imwrite(dst, medianImg)
        print("save to:"+dst)
        
    if filter_type==0 or filter_type==3:
        bilateralImg = cv.bilateralFilter(img, 5, 75, 75)
        if show_flag:
            cv.imshow("bilateralImg",bilateralImg)
        if PSNR_flag:
            ans = get_PSNR(Bscan, bilateralImg)
            print("PSNR of bilateral:"+str(ans))
        dst = get_dst(dst_path, "bilateral", index)
        cv.imwrite(dst, bilateralImg)
        print("save to:"+dst)

    if show_flag:
        key = cv.waitKey(0)
        if key == 27: #按esc键时，关闭所有窗口
            print("quit")
            cv.destroyAllWindows()

