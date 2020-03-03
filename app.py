from glob import glob

import cv2
import numpy as np
import pickle
import os

import saveMask
import ExtractTextureFeaturesAndMorphologicalFeatuures
import MorphologicalFeatureExtract
import ExtractTextureFeatures
from sklearn.externals import joblib
import segmentLungsOpenCV
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

# TensorFlow and tf.keras
import tensorflow as tf
# Flask
from flask import Flask, request, render_template, jsonify, send_from_directory
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image

from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

# Model saved with Keras model.save()
MODEL_PATH = 'models/3-conv-128-layer-dense-1-out-2-softmax-categorical-cross-2-CNN.h5'

# Load your own trained model
model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()  # Necessary
print('Model loaded. Start serving...')


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def prepareCNN(file_path):
    print("..........................file path ")
    print(file_path)
    IMGSIZE = 100
    Img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / 255
    new_array = cv2.resize(Img_array, (IMGSIZE, IMGSIZE))

    return new_array.reshape(-1, IMGSIZE, IMGSIZE, 1)


def prepareSVM(listOfParams):
    SVM_PATH = 'models/finalized_model.sav'

    # load SVM
    loaded_model = pickle.load(open(SVM_PATH, 'rb'))
    result = loaded_model.predict(listOfParams)
    # prob = loaded_model.predict_proba(listOfParams)
    return result


def model_predict(img, model):
    img = img.resize((100, 100))
    # IMGSIZE = 100
    # Img_array = cv2.imread(img,cv2.IMREAD_GRAYSCALE)/255
    # new_array = cv2.resize(Img_array,(IMGSIZE,IMGSIZE))
    # converted_array = new_array.reshape(-1,IMGSIZE,IMGSIZE,1)

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')
    print('###################################')
    print(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        print(img)

        # Save the image to ./uploads
        img.save("./uploads/image.jpeg")
        # img.save("./uploads/image.png")

        #save manually segmented mask---------------------------------
        segmentLungsOpenCV.segmentLung('./uploads/image.jpeg')
        # segmentManuallyImage = segmentLungsOpenCV.segmentLung('./uploads/image.jpeg')
        # gray1 = np.array(segmentManuallyImage).astype(np.uint8)
        # cv2.imwrite('static/SaveSegmentedLungOPenCV/Segmented_lung_openCV.jpeg', gray1)

        # save mask ----------------------------------------------------------------------------
        SegmentMaskModelPath = 'models/SENTAURS_Unet_Segment.h5'
        segment_mask_Model = load_model(SegmentMaskModelPath,
                                        custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

        # SEGMENTATION_CUSTOM_TEST_DIR = './uploads/image.jpeg'
        NEED_OF_SEGMENTATION_DIR = os.path.join("./")
        SEGMENTATION_CUSTOM_TEST_DIR = 'uploads'
        custom_test_files = glob(os.path.join(SEGMENTATION_CUSTOM_TEST_DIR, "*.jpeg"))

        NEED_OF_SAVE_DIR = os.path.join('static/SaveMask')
        # SAVE_CUSTOM_TEST_DIR = os.path.join(NEED_OF_SAVE_DIR, "1")

        test_gen = saveMask.test_generator_changed('uploads/image.jpeg', target_size=(512, 512))

        results = segment_mask_Model.predict_generator(test_gen, 1)
        print(results)
        kk = saveMask.save_result(NEED_OF_SAVE_DIR, results, custom_test_files)
        print(kk)

        try:
            os.remove('static/SaveMask/segmented_mask.jpeg')
        except:
            pass
        cv2.imwrite('static/SaveMask/segmented_mask.jpeg', kk)

        # ExtractTextureFeaturesAndMorphologicalFeatuures.extract_for_knn('uploads/image.jpeg','static/SaveMask/segmented_mask.jpeg')

        # # ---------------------------------------------------------------------------------------------------------------------------
        # #load random forest
        # # load model
        RandomForestHaralikTextureFeaturesModel = joblib.load('models/final1.pkl', mmap_mode='r')
        matrix = ExtractTextureFeaturesAndMorphologicalFeatuures.extract_for_randomForestt('uploads/image.jpeg','static/SaveMask/segmented_mask.jpeg')
        # get prediction given features
        print(matrix)
        prediction_prob_randomforest =[]
        pred_randomforest =[]
        prediction_prob_randomforest = RandomForestHaralikTextureFeaturesModel.predict_proba(matrix)
        pred_randomforest = RandomForestHaralikTextureFeaturesModel.predict(matrix)
        print(prediction_prob_randomforest)
        print(pred_randomforest)
        pred_result_randomForest = ''
        if(pred_randomforest[0]==0):
            pred_result_randomForest = "NORMAL"
        if (pred_randomforest[0] == 1):
            pred_result_randomForest = "PNEUMONIA"
        if (pred_randomforest[0] == 2):
            pred_result_randomForest = "TUBERCULOSIS"

        prediction_prob_randomforest = "{:.3f}".format(np.amax(prediction_prob_randomforest))

        # if(prediction_prob_randomforest< "{:.3f}".format(0.5)):
        #     pred_result_randomForest = "Hard to Predict"


        #-------------------------------------------

        #-----------------------------------------------------------------------------------
        # load KNN haralick features
        KNNHaralikTextureFeaturesModel = joblib.load('models/Texture+MorpolagicalFeatureFeedKNNModelFinal.pkl',mmap_mode='r')
        matrixKNN = ExtractTextureFeaturesAndMorphologicalFeatuures.extract_for_knn('uploads/image.jpeg','static/SaveMask/segmented_mask.jpeg')
        # get prediction given features
        print((matrixKNN))
        prediction_prob_KNN = KNNHaralikTextureFeaturesModel.predict_proba(matrixKNN)
        pred_KNN = KNNHaralikTextureFeaturesModel.predict(matrixKNN)
        print(prediction_prob_KNN)
        print(pred_KNN)
        pred_result_KNN = ''
        if (pred_KNN[0] == 0):
            pred_result_KNN = "NORMAL"
        if (pred_KNN[0] == 1):
            pred_result_KNN = "PNEUMONIA"
        if (pred_KNN[0] == 2):
            pred_result_KNN = "TUBERCULOSIS"

        prediction_prob_KNN = "{:.3f}".format(np.amax(prediction_prob_KNN))

        # if(prediction_prob_KNN<"{:.3f}".format(0.5)):
        #     pred_result_KNN = " Hard to Predict"


        # # -----------------------------------------------------------------------------------



        #---------------------------------------------------------------------------------------------------------------------------

        # Load Normal training model trained model
        MODEL_PATH_NORMAL = 'models/SENTAURS_MODIFIED_CNN_FOR_NON_SEGMENTED.h5'
        model = tf.keras.models.load_model(MODEL_PATH_NORMAL)
        testing_normal = model.predict([prepareCNN('./uploads/image.jpeg')])
        print('testing........................................................')
        print(testing_normal)
        result_normal = "Undefined"

        CATEGORIES = ["NORMAL", "PN", "TB"]
        print([float(testing_normal[0][0])])
        print([float(testing_normal[0][1])])
        print([float(testing_normal[0][2])])
        # print(CATEGORIES[int(prediction[0][0])])
        if float(testing_normal[0][0]) > 0.5 or float(testing_normal[0][1]) > 0.5 or (testing_normal[0][2]) > 0.5:
            if float(testing_normal[0][0]) > float(testing_normal[0][1]) and (testing_normal[0][0]) > float(
                    testing_normal[0][2]):
                print("NORMAL")
                result_normal = "NORMAL"
            elif float(testing_normal[0][1]) > float(testing_normal[0][2]):
                print("PNEUMONIA")
                result_normal = "PNEUMONIA"
            elif float(testing_normal[0][2]) > float(testing_normal[0][1]):
                print("TUBERCULOSIS")
                result_normal = "TUBERCULOSIS"

        else:
            print("UnDefined")
            result = "UnDefined"

        pred_proba_normal = "{:.3f}".format(np.amax(testing_normal))

        # Load SEGMENTED training model trained model
        MODEL_PATH_SEGMENTED = 'models/SENTAURS_MODIFIED_CNN_FOR_MODEL_SEGMENTED.h5'
        model_segmented = tf.keras.models.load_model(MODEL_PATH_SEGMENTED)
        testing_segmented = model_segmented.predict([prepareCNN('static/SaveMask/segmented_mask.jpeg')])
        print('testing_Segmented........................................................')
        print(testing_segmented)
        result_segmented = "Undefined"

        CATEGORIES = ["NORMAL", "PN", "TB"]
        print([float(testing_segmented[0][0])])
        print([float(testing_segmented[0][1])])
        print([float(testing_segmented[0][2])])
        # print(CATEGORIES[int(prediction[0][0])])
        if float(testing_segmented[0][0]) > 0.5 or float(testing_segmented[0][1]) > 0.5 or (
                testing_segmented[0][2]) > 0.5:
            if float(testing_segmented[0][0]) > float(testing_segmented[0][1]) and (testing_segmented[0][0]) > float(
                    testing_segmented[0][2]):
                print("NORMAL")
                result_segmented = "NORMAL"
            elif float(testing_segmented[0][1]) > float(testing_segmented[0][2]):
                print("PNEUMONIA")
                result_segmented = "PNEUMONIA"
            elif float(testing_segmented[0][2]) > float(testing_segmented[0][1]):
                print("TUBERCULOSIS")
                result_segmented = "TUBERCULOSIS"

        else:
            print("UnDefined")
            result_segmented = "UnDefined"

        pred_proba_segmented = "{:.3f}".format(np.amax(testing_segmented))

        # Load SEGMENTED Manually training model trained model
        MODEL_PATH_SEGMENTED_MANUALLY = 'models/SENTAURS_MODIFIED_CNN_FOR_MANUALLY_SEGMENTED.h5'
        model_segmented_manually = tf.keras.models.load_model(MODEL_PATH_SEGMENTED_MANUALLY)
        testing_segmented_manually = model_segmented_manually.predict([prepareCNN('static/SaveSegmentedLungOPenCV/Segmented_lung_openCV.jpeg')])
        print('testing_Segmented_manually........................................................')
        print(testing_segmented_manually)
        result_segmented_manually = "Undefined"

        CATEGORIES = ["NORMAL", "PN", "TB"]
        print([float(testing_segmented_manually[0][0])])
        print([float(testing_segmented_manually[0][1])])
        print([float(testing_segmented_manually[0][2])])
        # print(CATEGORIES[int(prediction[0][0])])
        if float(testing_segmented_manually[0][0]) > 0.5 or float(testing_segmented_manually[0][1]) > 0.5 or (
                testing_segmented_manually[0][2]) > 0.5:
            if float(testing_segmented_manually[0][0]) > float(testing_segmented_manually[0][1]) and (testing_segmented_manually[0][0]) > float(
                    testing_segmented_manually[0][2]):
                print("NORMAL")
                result_segmented_manually = "NORMAL"
            elif float(testing_segmented_manually[0][1]) > float(testing_segmented_manually[0][2]):
                print("PNEUMONIA")
                result_segmented_manually = "PNEUMONIA"
            elif float(testing_segmented_manually[0][2]) > float(testing_segmented_manually[0][1]):
                print("TUBERCULOSIS")
                result_segmented_manually = "TUBERCULOSIS"

        else:
            print("UnDefined")
            result_segmented_manually = "UnDefined"

        pred_proba_segmented_manually = "{:.3f}".format(np.amax(testing_segmented_manually))

        return jsonify(result=result_normal, probability=pred_proba_normal, result_no_rotated=result_segmented,
                       probability_no_rotated=pred_proba_segmented,result_segmented_manually=result_segmented_manually,pred_proba_segmented_manually=pred_proba_segmented_manually,
                       pred_result_randomForest=pred_result_randomForest,prediction_prob_randomforest=prediction_prob_randomforest,
                       pred_result_KNN=pred_result_KNN,prediction_prob_KNN=prediction_prob_KNN)
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
