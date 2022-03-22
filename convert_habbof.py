import os
import shutil
import cv2
import math

habbof_dir = "/home/long/Documents/HABBOF/"
folders = ['Lab1', 'Lab2', 'Meeting1', 'Meeting2']


for fd in folders:
    folder = habbof_dir + fd
    for f in os.listdir(folder):
        if '.jpg' in f:
            continue
        img_path = os.path.join(folder, f.replace('txt', 'jpg'))
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        reader = open(os.path.join(folder,f), 'r')
        lines = reader.readlines()
        writer = open(os.path.join(habbof_dir, 'labels', fd+'_'+f), 'w')
        for line in lines:
            _, cx, cy, w, h, a = line.split(' ')
            cx = int(cx)/width
            cy = int(cy)/height
            w = int(w)/width
            h = int(h)/height
            r = math.cos(math.radians(int(a)))
            i = math.sin(math.radians(int(a)))
            writer.write("0 {} {} {} {} {} {}\n". format(cx, cy, w, h, r, i))
        writer.close()
        reader.close()
        shutil.copy(img_path, os.path.join(habbof_dir, 'images',fd+'_'+ f.replace('txt', 'jpg')))
        
    
