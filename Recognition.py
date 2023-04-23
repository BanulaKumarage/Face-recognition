from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os

# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 


class FaceRecognition:

    # Extract the images from img_path and create the data.pt
    @staticmethod
    def extract(img_path = 'Images'):
        # Read data from folder
        dataset = datasets.ImageFolder(img_path) # Images folder path 
        idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

        def collate_fn(x):
            return x[0]

        loader = DataLoader(dataset, collate_fn=collate_fn)

        name_list = [] # list of names corrospoing to cropped Images
        embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

        for img, idx in loader:
            face, prob = mtcnn0(img, return_prob=True) 
            if face is not None and prob>0.92:
                emb = resnet(face.unsqueeze(0)) 
                embedding_list.append(emb.detach()) 
                name_list.append(idx_to_class[idx])        

        # save data
        data = [embedding_list, name_list] 
        torch.save(data, 'data.pt') # saving data.pt file

    #Recognize faces with webcam with the help of data.pt
    @staticmethod
    def recognize():    
        # loading data.pt file
        load_data = torch.load('data.pt')
        embedding_list = load_data[0]
        name_list = load_data[1] 

        print ("Streaming started")
        cam = cv2.VideoCapture(1) 
        while True:
            ret, frame = cam.read()
            if not ret:
                print("fail to grab frame, try again")
                break

            img = Image.fromarray(frame)
            img_cropped_list, prob_list = mtcnn(img, return_prob=True) 

            if img_cropped_list is not None:
                boxes, _ = mtcnn.detect(img)

                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 

                        dist_list = [] # list of matched distances, minimum distance is used to identify the person

                        for emb_db in embedding_list:
                            dist = torch.dist(emb, emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list) # get minumum dist value
                        min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                        name = name_list[min_dist_idx] # get name corrosponding to minimum dist

                        box = boxes[i] 
                        box = box.astype(int)

                        original_frame = frame.copy() # storing copy of frame before drawing on it

                        if min_dist >= 0.90:
                            name= 'Unknown'

                        frame = cv2.putText(
                            frame,
                            f'{name}',
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        box = box.astype(int)
                        frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)

            cv2.imshow("Facial Recognition", frame)

            k = cv2.waitKey(1)
            if k%256==27: # ESC
                print('Esc pressed, closing...')
                break        

        cam.release()
        cv2.destroyAllWindows()


while True:
    mode = int(input("1-Extract the image data\n2-Recogition system\n0-Exit\n"))
    
    if mode == 1:
        print ("Extracting data")
        FaceRecognition.extract()
        print ("Data is extracted")
    elif mode == 2:
        FaceRecognition.recognize()
    elif mode == 0:
        print ("Exiting from system")
        break
