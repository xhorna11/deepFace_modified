import face_recognition
import os
from os import listdir
from os.path import isfile, join
import sys
import time
import cv2
import numpy
import array
import glob
import errno

def getReferences(dir):
    subdirs=listdir(dir) 
    names=[];
    emb=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            image = face_recognition.load_image_file(file)
            encodings=face_recognition.face_encodings(image)
            if (encodings):
                names.append(name)
                emb.append(encodings[0])
    return emb,names



'''
#Vypocet 128 bodov na tvary a nasledne ulozenie do textoveho suboru pre kazdu osobu.

def main(args,distance):
    subdirs=listdir(args) 
    features=[];
    names=[];
    
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        txtFile = open('C:/Users/Veronika/Desktop/video/Reference/' + name + '/' + name + '.txt', 'w') 
        for file in (files):
            file=join(subdir,file);
            image = face_recognition.load_image_file(file)
            align=face_recognition.face_landmarks(image)
            encodings=face_recognition.face_encodings(image)
            for encoding in (encodings):
                output = encoding
                txtFile.write("%s\n" % output)
            print("File", file)
            print("Original value: ", output)
        
if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/video/Reference",0.55)) or 0)

'''

'''
#Vypocet vah pre kazdu osobu a ulozenie do textoveho suboru.

def main(args,distance):
    subdirs=listdir(args) 
    results = []
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        txtFileStd = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'Smerodajna_odchylka_LFW.txt', 'r')
        text=txtFileStd.read()
        text=text.replace("]", " ")
        text=text.replace("[", "[ ")
        result=text.split()
        first=True
        idx=0;
        for i,r in enumerate(result):
            if (r=='['):
                f=[]
                idx=idx+1
                for n in range(i+1,i+129):
                    f.append(float(result[n]))  
                if (first):
                    features=numpy.array(f)
                    first=False
                else:
                    features=numpy.vstack([features,f])
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        txtFile = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'Smerodajna_odchylka.txt', 'r')
        txtFileWeight = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'Vaha.txt', 'w')
        text=txtFile.read()
        text=text.replace("]", " ")
        text=text.replace("[", "[ ")
        resultPersonal=text.split()
        first=True
        idx=0;
        for i,r in enumerate(resultPersonal):
            if (r=='['):
                f=[]
                idx=idx+1
                for n in range(i+1,i+129):
                    f.append(float(resultPersonal[n]))  
                if (first):
                    featuresPersonal=numpy.array(f)
                    first=False
                else:
                    featuresPersonal=numpy.vstack([features,f])
        
        txtFileWeight.write("%s\n" % str(features/(featuresPersonal + 1)))
        
       
if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/video/Reference",0.55)) or 0)


'''



'''
#Normalizovanie vah pre kazdu osobu a ulozenie do textoveho suboru.
def main(args,distance):
    subdirs=listdir(args) 
    results = []
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        txtFileStd = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'Vaha.txt', 'r')
        txtFileNorm = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'NormVaha.txt', 'w')
        text=txtFileStd.read()
        text=text.replace("]", " ")
        text=text.replace("[", "[ ")
        result=text.split()
        first=True
        idx=0;
        for i,r in enumerate(result):
            if (r=='['):
                f=[]
                idx=idx+1
                for n in range(i+1,i+129):
                    f.append(float(result[n]))  
                if (first):
                    features=numpy.array(f)
                    first=False                                  
                else:
                    features=numpy.vstack([features,f])

        number=numpy.sum(features, axis=0)
        const=128/number
        txtFileNorm.write("%s\n" % str(numpy.array(numpy.multiply(features, const))))
       
if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/video/Reference",0.55)) or 0)
'''





'''
#Vypocet priemeru, medianu a smerodajnej odchylky pre kazdu osobu a ulozenie do textoveho suboru.

def main(args,distance):
    subdirs=listdir(args) 
    results = []
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        txtFileAvg = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'priemer.txt', 'w')
        txtFileMedian = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'median.txt', 'w')
        txtFileStd = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + 'Smerodajna_odchylka.txt', 'w')
        txtFile = open('C:/Users/Veronika/Desktop/video/Reference/' + name +'/' + name + '.txt', 'r')
        text=txtFile.read()
        text=text.replace("]", " ")
        text=text.replace("[", "[ ")
        result=text.split()
        first=True
        idx=0;
        for i,r in enumerate(result):
            if (r=='['):
                f=[]
                idx=idx+1
                for n in range(i+1,i+129):
                    f.append(float(result[n]))  
                if (first):
                    features=numpy.array(f)
                    first=False
                else:
                    features=numpy.vstack([features,f])
        std=numpy.std(features, axis=0)
        avg=numpy.average(features, axis=0)
        for i in range(len(features)):
            if (i==0):
                std_lfw=numpy.power(features[i,:]-avg,2)
            else:
                std_lfw+=numpy.square(features[i,:]-avg)
        std_lfw=numpy.sqrt(std_lfw/len(features))

        txtFileAvg.write("%s\n" % str(numpy.average(features, axis=0)))
        txtFileMedian.write("%s\n" % str(numpy.median(features,axis=0)))
        txtFileStd.write("%s\n" % str(numpy.std(features, axis=0)))
        avg_features=numpy.average(features,axis=0)
        med_features=numpy.median(features,axis=0)
        std_features=numpy.std(features,axis=0)
        print(avg_features)
        print(med_features)
        print(std_features)

        
       
if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/video/Reference",0.55)) or 0)
'''


'''
#Vypocet priemeru, smerodajnej odchylky a 128 bodov na tvary pre databazu LFW a nasledne ulozenie do textoveho suboru.

def main(args,distance):
    subdirs=listdir(args) 
    results = []    
    txtFileAvg = open('C:/Users/Veronika/Documents/Visual Studio 2015/Projects/facenet_recognition - kopie/facenet_recognition/align/datasets/lfw/priemer.txt', 'w')
    txtFileStd = open('C:/Users/Veronika/Documents/Visual Studio 2015/Projects/facenet_recognition - kopie/facenet_recognition/align/datasets/lfw/Smerodajna_odchylka.txt', 'w')
    txtFile = open('C:/Users/Veronika/Documents/Visual Studio 2015/Projects/facenet_recognition - kopie/facenet_recognition/align/datasets/lfw/Data.txt', 'r')
    text=txtFile.read()
    text=text.replace("]", " ")
    text=text.replace("[", "[ ")
    result=text.split()
    first=True
    idx=0;
    for i,r in enumerate(result):
        if (r=='['):
            f=[]
            idx=idx+1
            for n in range(i+1,i+129):
                f.append(float(result[n]))  
            if (first):
                features=numpy.array(f)
                first=False
            else:
                features=numpy.vstack([features,f])
    std=numpy.std(features, axis=0)
    avg=numpy.average(features, axis=0)
    for i in range(len(features)):
        if (i==0):
            std_lfw=numpy.power(features[i,:]-avg,2)
        else:
            std_lfw+=numpy.square(features[i,:]-avg)
    std_lfw=numpy.sqrt(std_lfw/len(features))

    txtFileAvg.write("%s\n" % str(numpy.average(features, axis=0)))
    txtFileStd.write("%s\n" % str(numpy.std(features, axis=0)))
    avg_features=numpy.average(features,axis=0)
    med_features=numpy.median(features,axis=0)
    std_features=numpy.std(features,axis=0)
    print(avg_features)
    print(std_features)

        
       
if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Documents/Visual Studio 2015/Projects/facenet_recognition - kopie/facenet_recognition/align/datasets/lfw/",0.55)) or 0)
'''








#Testovanie s databazou videi.

def getReferences(dir):
    subdirs=listdir(dir) 
    names=[];
    emb=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        txtFile = open(subdir + '/priemer.txt', 'r')
        text=txtFile.read()
        text=text.replace("]", "")
        text=text.replace("[", "")
        result=text.split()
        f=[]
        for n in range(0,128):
            f.append(float(result[n]))
        features=numpy.array(f)
        emb.append(features)
        names.append(name)
        txtFile.close()
    return emb, names



def proccessVideo(dir,dist):
    [references,names]=getReferences(dir+"reference")
    dir=dir+"test"
    subdirs=listdir(dir) 
    TP=0
    FP=0
    FN=0
    TN=0
    total=0
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        txtFile = open('C:/Users/Veronika/Desktop/video/Reference/' + name + '/NormVaha.txt', 'r')
        text=txtFile.read()
        text=text.replace("]", " ")
        text=text.replace("[", "[ ")
        result=text.split()
        first=True
        idx=0;
        for i,r in enumerate(result):
            if (r=='['):
                f=[]
                idx=idx+1
                for n in range(i+1,i+129):
                    f.append(float(result[n]))  
                if (first):
                    weights=numpy.array(f)
                    first=False
                else:
                    weights=numpy.vstack([weights,f])
        
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            videoCapture=cv2.VideoCapture(file)
            if (not videoCapture.isOpened()):
                continue
            embeddings=[]
            ret,frame=videoCapture.read()
            while(ret):
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                encodings=face_recognition.face_encodings(image)
                if (encodings):
                    embeddings.append(encodings[0])
                ret,frame=videoCapture.read()
            for n, tst in enumerate(embeddings):
                face_distances = face_recognition.face_distance(references, tst, weights)
                for i, face_distance in enumerate(face_distances):
                    if (face_distance<dist):
                        if (name==names[i]):
                            TP+=1      
                        else:
                            FP+=1
                    else:
                        if (name==names[i]):
                            FN+=1      
                        else:
                            TN+=1
                    total=total+1
            print("TP=",TP," FP=",FP," FN=",FN," TN=",TN," Total=",total)

def main(args,distance):
    proccessVideo(args,distance)
    return 0
   

if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/video/",0.55)) or 0)
