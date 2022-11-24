def getBasePath():
    import os
    return os.path.dirname(os.path.abspath(__file__))


def createFileOrFolderPath(base_path, file_or_folder_name):
    import os
    return os.path.join(base_path, file_or_folder_name)


def getFolderContent(folder_path):
    import os
    return os.listdir(folder_path)


def chooseImagesFormats():
    raster_imgs_formats = ["jpeg", "jpg", "png", "gif", "bmp", "webp"]

    while True:
        decision = input("voulez vous utilisez ces formats \"Raster\" pour vos images? validez par 'o': ")
        if decision.lower() == 'o':
            print("formats enregistrés.")
            return raster_imgs_formats
        else:
            imgs_formats = []
            while True:
                format = input("entrez un format d'image (jpg, png, etc): ")
                imgs_formats.append(format)
                decision = input("appuyez sur 'q' pour quitter? ")
                if decision.lower() == 'q':
                    print("formats enregistrés.")
                    return imgs_formats


def setImageSize():
    print("- entrez les dimensions d'une image \"Portrait\" avec un rapport de 3:4 -")
    print("exemples: 384*512, 480*640, 600*800, 768*1024, etc")
    while True:
        width = int(input("entrez la largeur de l'image: "))
        height = int(input("entrez la hauteur de l'image: "))
        if(width < height) and (width / height == 3 / 4):
            break
        else:
            print("la largeur d'une image Portrait ne peut pas etre supérieure à sa hauteur!")
            print("le rapport entre la largeur de l'image sur sa hauteur n'est pas égale à 3:4!")
    return width, height


def setScaleFactorAndMinNeighbors():
    print("entrez les parametres \"scaleFactor\" et \"minNeighbors\" de la fct \"detectMultiScale()\":")
    scaleFactor = float(input("scaleFactor (type \"float\") (ex: 1.3): "))
    minNeighbors = int(input("minNeighbors (type \"int\") (ex: 3): "))
    return scaleFactor, minNeighbors


def readImage(img_path):
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return img


def resizeImage(img, img_size):
    import cv2
    # all imgs must be Portrait 3:4
    img_resized = cv2.resize(img, img_size)
    return img_resized


def convertImageToGrayscale(img):
    # Convert an image to grayscale
    import cv2

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grayscale


def writeImage(img_path, img):
    import cv2

    cv2.imwrite(img_path, img)
    print("image successfully written in: {}".format(img_path))


def displayImageInWindow(img_name, img):
    # Display an image in a window
    import cv2

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def verifyImageFormat(file, imgs_formats):
    for format in imgs_formats:
        if file.lower().endswith(format):
            return True
        else:
            continue
    return False


def getImageLabel(file_path):
    # the label of an image is the name of the folder where the image is located
    import os
    return os.path.basename(file_path).replace(" " or "_", "-").lower()  # replace any space ' ' or underscore '_' with a dash '-' in the label's name and put it in lowercase


def extractFacesAndLabelsFromDataset(dataset_path, imgs_formats, img_size, classifier_path, scaleFactor, minNeighbors):
    import os
    import cv2

    classifier = cv2.CascadeClassifier(classifier_path)
    id = 0  #id is a number that identifies an image's label
    labels_ids = {}
    X = []
    Y = []

    for base_path, folders, files in os.walk(dataset_path):
        # print(base_path)
        for file in files:
            # print(file)
            if verifyImageFormat(file, imgs_formats):
                img_path = createFileOrFolderPath(base_path, file)
                img_label = getImageLabel(base_path)

                if not img_label in labels_ids:
                    labels_ids[img_label] = id
                    id += 1
                id_ = labels_ids[img_label]
                # print(labels_ids)
                # print("id_: ", id_)

                img_original = readImage(img_path)
                img_resized = resizeImage(img_original, img_size)
                img_grayscale = convertImageToGrayscale(img_resized)

                # Detect the face on the picture and its coordinates with the detectMultiscale() fct
                detected_faces = classifier.detectMultiScale(img_grayscale, scaleFactor, minNeighbors)
                # print(detected_faces)

                if len(detected_faces) == 0:  # If we detect no face in the image
                    print("Il n'y a aucun visage sur la photo {}".format(img_path))
                elif len(detected_faces) > 1:  # If we detect more than one face in the image
                    print("Il y a plus qu'un visage sur la photo {}".format(img_path))
                else:  # If we detect one face in the image..

                    # ..we extract the detected face from the picture
                    for (x, y, width, height) in detected_faces:
                        detected_face = img_grayscale[y: y + height, x: x + width]
                        # print("detected_face:", detected_face)
                        # print("detected_face type:", detected_face.dtype)   #numpy array with a "unint8" type

                        # Display the detected face in a window
                        # displayImageInWindow(str(img_label), detected_face)
                        # print("img label:", img_label)

                        X.append(detected_face)
                        Y.append(id_)
            else:
                print("le format de l'image {} est incompatible!".format(file))
    return X, Y, labels_ids


def extractFaceCoordinates(detected_face_coordinates):
    (x, y, width, height) = detected_face_coordinates
    return x, y, width, height


def saveDataInPickleFile(dictio, pickle_file_path):
    import pickle

    with open(pickle_file_path, "wb") as file:
        pickle.dump(dictio, file)
        print("pickle file successfully saved in: {}".format(pickle_file_path))


def readDataFromPickleFile(pickle_file_path):
    import pickle

    with open(pickle_file_path, "rb") as file:
        dictio = pickle.load(file)
        print("pickle file successfully loaded from: {}".format(pickle_file_path))
    return dictio


def instanciateFaceRecognizer():
    import cv2

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    return recognizer


def trainFaceRecognizer(X, Y, trained_recognizer_file_path):
    import numpy as np

    recognizer = instanciateFaceRecognizer()
    Y = np.array(Y) #convert a list to a numpy array
    recognizer.train(X, Y)
    print("recognizer successfully trained.")
    recognizer.save(trained_recognizer_file_path)  #save the trained data in a YML file
    print("recognizer successfully saved in: {}".format(trained_recognizer_file_path))


def extractFaceFromImage(img_path, imgs_formats, img_size, classifier_path, scaleFactor, minNeighbors, detected_face_file_path):
    import cv2

    classifier = cv2.CascadeClassifier(classifier_path)

    if verifyImageFormat(img_path, imgs_formats):

        img_original = readImage(img_path)
        img_resized = resizeImage(img_original, img_size)
        img_grayscale = convertImageToGrayscale(img_resized)

        # Display the image in a window
        # displayImageInWindow(str(img_path), img_final)

        # Detect the face on the image and its coordinates with the detectMultiscale() fct
        detected_faces = classifier.detectMultiScale(img_grayscale, scaleFactor, minNeighbors)

        if len(detected_faces) == 0:  # If we detect no face in the training image
            print("we found no faces in {}".format(img_path))
        elif len(detected_faces) > 1:  # If we detect more than one face in the training image
            print("we found more than one face in {}".format(img_path))
        else:  # If we detect one face in the training image (len(detected_faces) == 1)
            for (x, y, width, height) in detected_faces:
                # print(x, y, width, height)
                detected_face_coordinates = (x, y, width, height)
                detected_face = img_grayscale[y: y + height, x: x + width]
                # print(detected_face)
                detected_face_with_colors = img_resized[y: y + height, x: x + width]
                # displayImageInWindow(img_path, detected_face_with_colors)

                # Save the detected face in a file
                writeImage(detected_face_file_path, detected_face)

                return detected_face, detected_face_coordinates
    else:
        print("le format de l'image {} est incompatible!".format(img_path))


def readFaceRecognizerTrainedData(trained_recognizer_file_path):
    recognizer = instanciateFaceRecognizer()
    recognizer.read(trained_recognizer_file_path)
    return recognizer


def runPredictionOnImage(trained_recognizer_file_path, img_path):
    recognizer = readFaceRecognizerTrainedData(trained_recognizer_file_path)
    img = readImage(img_path)
    id, loss = recognizer.predict(img)   # id: label's id
    print("prediction on image {} done.".format(img_path))
    # print("image label_id: ", id)
    # print("loss : ", loss)
    # The higher the loss value is, the less similar the 2 faces are
    return id, loss


# reverse the key/value pair in a dictionnary:
# ex: from {"person_name": id} to {id: "person_name"}
def reverseKeyValuePairInDictio(dictio):
    reversed_dictio = {v: k for k, v in dictio.items()}
    return reversed_dictio


def getLabelFromId(key, dictio):
    if key in dictio:
        return dictio[key]
    else:
        print("la clé '{}' n'existe pas dans le dictionnaire!".format(key))


def drawFrameOnFace(img, face_coordinates):
    import cv2

    (x, y, width, height) = extractFaceCoordinates(face_coordinates)

    frame_color = (0, 255, 0)
    frame_stroke = 2

    cv2.rectangle(img, (x, y), (x + width, y + height), frame_color, frame_stroke)


def writeTextOnImage(img, face_coordinates, label, loss):
    import cv2

    (x, y, width, height) = extractFaceCoordinates(face_coordinates)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color_1 = (0, 255, 0)
    font_color_2 = (0, 0, 255)
    font_stroke = 2
    person_name = label.replace("-" or "_", " ")
    loss_value = str("loss = {:.3f}".format(loss))

    cv2.putText(img, person_name, (x, y-20), font, 1, font_color_1, font_stroke, cv2.LINE_AA)
    cv2.putText(img, loss_value, (x, 620), font, 1, font_color_2, font_stroke, cv2.LINE_AA)


def writeDataOnImage(img_path, img_size, face_coordinates, label, loss):

    img_original = readImage(img_path)
    img_resized = resizeImage(img_original, img_size)

    drawFrameOnFace(img_resized, face_coordinates)
    writeTextOnImage(img_resized, face_coordinates, label, loss)

    displayImageInWindow(label, img_resized)
