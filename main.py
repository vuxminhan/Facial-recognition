from pca.main import FaceRecognitionModel
import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

if __name__ == '__main__':
    clf = SVC(kernel='rbf', class_weight='balanced')
    #get the working dir
    working_dir = os.getcwd()
    face_model = FaceRecognitionModel(classifier=clf, data_path=os.getcwd()+"/pca/att_faces", n_components=50)
    face_model.load_data()
    face_model.split_data()
    face_model.perform_pca()
    face_model.project_on_eigenfaces()
    # face_model.plot_eigenfaces()
    face_model.print_classifier_info()
    face_model.train_classifier()
    face_model.evaluate_model()