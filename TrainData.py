import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def getImageWithId(path):
    
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        
        print(faceNp)
        print(os.path.split(imagePath)[1].split('.'))
        #split to get ID of the image
        ID=int(os.path.split(imagePath)[1].split('.')[1])
        
        faces.append(faceNp) 
        IDs.append(ID)
        
        cv2.imshow('tranning', faceNp)
        cv2.waitKey(10)

    return IDs, faces

Ids,faces=getImageWithId(path)

# trainning
recognizer.train(faces,np.array(Ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
 

