import os
import argparse
import csv
from skimage import io
from pprint import pprint
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector
from PIL import Image
from tqdm import tqdm
import pandas as pd
import re
import numpy as np


TOP_K_MATCH = 1


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def process_image(path):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    return face_recognizer.perform(face_images)


def preprocess_image(image, image_size):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(pil_image)


def extract_celebrity_name(text):
    separator = r"[ _]"
    patterns = [
        rf"^A{separator}portrait{separator}of{separator}(.+?)_(\d+)\.png$",
        rf"^An{separator}image{separator}capturing{separator}(.+?){separator}at{separator}a{separator}public{separator}event_(\d+)\.png$",
        rf"^An{separator}oil{separator}painting{separator}of{separator}(.+?)_(\d+)\.png$",
        rf"^A{separator}sketch{separator}of{separator}(.+?)_(\d+)\.png$",
        rf"^(.+?){separator}in{separator}an{separator}official{separator}photo_(\d+)\.png$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1).replace("_", " ")

    raise ValueError("The input image name does not match any of the expected patterns: " + text)


if __name__ == '__main__':
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description='Inference script for Giphy Celebrity Classifier model')
    parser.add_argument('--image_folder', type=str, help='path or link to the image folder', default=None)
    parser.add_argument('--save_excel_path', type=str, help='path to save the excel file', default=None)
    parser.add_argument('--results-csv', type=str, help='path to write per-image GCD results', default=None)

    args = parser.parse_args()

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

    model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))

    face_detector = FaceDetector(
        os.getenv('APP_DATA_DIR'),
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA') == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv('APP_DATA_DIR'),
        use_cuda=os.getenv('USE_CUDA') == "true",
        top_n=5 
    )

    image_files=os.listdir(args.image_folder)
    image_names=sorted(image_files)   #sort image files
    
    predictions_list=[]
    p_celebrity_list=[]  
    n_no_faces=0
    result_rows=[]
    
    for file in tqdm(image_names):
        image_path=os.path.join(args.image_folder,file)
        
        predictions = process_image(image_path) # precdictions contain the probabilities of the top n celebrities for one image
        if len(predictions)==0:     # if no face detected
            n_no_faces+=1
            p_celebrity_list.append('N')  # give empty string if no face detected
            predictions_list.append([None] * 5)
            result_rows.append({
                'filename': file,
                'expected_name': extract_celebrity_name(file),
                'top1_name': '',
                'face_detected': '0',
                'correct': '0',
            })
        else:
            predictions_new_label=[]
            for prediction in predictions[0][0]:
                celebrity_label, prob=prediction
                celebrity_label=str(celebrity_label)  
                # Modify label format
                celebrity_name=celebrity_label.split('_[',1)[0].replace('_',' ')
                prediction=(celebrity_name,prob)
                predictions_new_label.append(prediction)
            predictions_list.append(predictions_new_label)

            print('************************')
            print(predictions_new_label[0][0])
            expected_name = extract_celebrity_name(file)
            print(expected_name)
            matched_prediction = next(
                (
                    prediction
                    for prediction in predictions_new_label[:TOP_K_MATCH]
                    if prediction[0].lower() == expected_name.lower()
                ),
                None,
            )
            if matched_prediction is not None:   # if the target celebrity is in top-k predictions
                p_celebrity_list.append(matched_prediction[1])
            else:
                p_celebrity_list.append(0)   # if the target celebrity is absent from top-k predictions
            result_rows.append({
                'filename': file,
                'expected_name': expected_name,
                'top1_name': predictions_new_label[0][0],
                'face_detected': '1',
                'correct': '1' if matched_prediction is not None else '0',
            })
    print('-------------------')
    print('Total number of images with no faces detected:', n_no_faces)           

    # save as excel file
    df=pd.DataFrame(predictions_list, columns=['top1','top2','top3','top4','top5'])
    df.index=image_names
    df['p_celebrity_correct']=p_celebrity_list
    print('-------------------')
    print('Given face detected, the celebrity classification accuracy is:')

    # Calculate the number of non-zero and non-N values in p_celebrity_list and then divided by the number of non-N values.
    detected_face_count = sum(1 for prediction in p_celebrity_list if prediction != 'N')
    accuracy = (
        sum(1 for prediction in p_celebrity_list if prediction != 0 and prediction != 'N')
        / detected_face_count
        if detected_face_count
        else 0.0
    )
    print(f'GCD Accuracy: {accuracy:.2%}')

    if args.save_excel_path is not None:
        df.to_excel(args.save_excel_path, index=True)

    if args.results_csv is not None:
        with open(args.results_csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=['filename', 'expected_name', 'top1_name', 'face_detected', 'correct'],
            )
            writer.writeheader()
            writer.writerows(result_rows)
        
