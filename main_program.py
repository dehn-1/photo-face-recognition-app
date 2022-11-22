from functions import *

# 1. Accès et prétraitements sur les données du dataset (photos)

base_path = getBasePath()
dataset_imgs_folder_path = createFileOrFolderPath(base_path, "dataset_images")  # Chemin du dataset

# Controle sur les formats et sur la taille des photos du dataset et celles entrées par le user
# Controle sur les formats des photos (jpg, png, etc)
# imgs_formats = ["jpeg", "jpg", "png", "gif", "bmp", "webp"]
imgs_formats = chooseImagesFormats()

# Controle sur la taille (dimension) des images
img_size = setImageSize()  # taille standard de toutes les images
# img_size = (480, 640)


# 2. Définir le Modèle: tout le dataset
# les données d'entrée (X) du dataset sont les visages sur les images du dataset
# les données de sortie (Y) sont les labels des visages détectés sur les photos du dataset

print("-Traitements et Apprentissage sur les photos du dataset-")
# Saisie des params "scaleFactor" et "minNeighbors" pour les photos du dataset
# scaleFactor = 1.3
# minNeighbors = 3
scaleFactor, minNeighbors = setScaleFactorAndMinNeighbors()

# Chemin du classifieur
classifier_path = createFileOrFolderPath(base_path, "haarcascade_dataset/haarcascade_frontalface_default.xml")

# Extraire les données d'entrée (X) et les données de sortie (Y) du dataset
X, Y, labels_ids = extractFacesAndLabelsFromDataset(dataset_imgs_folder_path, imgs_formats, img_size, classifier_path,
                                                    scaleFactor, minNeighbors)
# print("X: ", X)
# print("Y: ", Y)
# print("labels_ids: ", labels_ids)

# Enregister les labels (noms des personnes) des visages détectés sur les photos du dataset et leurs identifiants, dans un fichier "pickle"
saveDataInPickleFile(labels_ids, "labels.pickle")
pickle_file_path = createFileOrFolderPath(base_path, "labels.pickle")

# 3. Apprentissage du Modèle (training the recognizer)
trainFaceRecognizer(X, Y, "trained_recognizer.yml")
trained_recognizer_path = createFileOrFolderPath(base_path, "trained_recognizer.yml")
# output: trained_recognizer.yml = c'est le Modèle estimé


# 5. Prédiction:
print("-Prédictions sur les photos entrées par l'utilisateur-")
# Chemin des images entrées par le user
input_imgs_folder_path = createFileOrFolderPath(base_path, "input_images")
# print(input_imgs_folder_path)
input_imgs = getFolderContent(input_imgs_folder_path)
# print(input_imgs)

detected_faces_folder_path = createFileOrFolderPath(base_path, "detected_face")

# Saisie des params "scaleFactor" et "minNeighbors" pour les photos entrées par le user
# scaleFactor = 1.3
# minNeighbors = 3
scaleFactor, minNeighbors = setScaleFactorAndMinNeighbors()

i = 1
# Parcours et traitements sur chaque photo entrée par le user
for img in input_imgs:
    img_path = createFileOrFolderPath(input_imgs_folder_path, img)
    # print(img_path)

    detected_face_file_name = "detected_face_" + str(i) + ".png"
    i += 1

    detected_face_file_path = createFileOrFolderPath(detected_faces_folder_path, detected_face_file_name)

    # Extraire le visage de la personne sur la photo ainsi que les coordonnées de ce dernier, en se basant sur un classifieur "Haar cascade"
    detected_face, detected_face_coordinates = extractFaceFromImage(img_path, imgs_formats, img_size, classifier_path,
                                                                    scaleFactor, minNeighbors, detected_face_file_path)

    # Prédire de l'identité des personnes sur les photos, en se basant sur le Modèle estimé (trained_recognizer.yml)
    id, loss = runPredictionOnImage(trained_recognizer_path, detected_face_file_path)

    # Charger le fichier "pickle"
    labels_ids_dictio = loadDataFromPickleFile(pickle_file_path)

    # inverser la clé et la valeur dans le dictionnaire contenant les labels (noms des personnes) et leurs identifiants
    ids_labels_dictio = reverseKeyValuePairInDictio(labels_ids_dictio)
    # print("ids_labels_dictio: ", ids_labels_dictio)

    # Extraire les labels (noms des personnes) à partir de leurs identifiants (entier)
    label = getLabelFromId(id, ids_labels_dictio)
    # print(label)

    # 6. Afficher la photo entrée par le user dans une fenetre et écrire les données de prédiction sur cette dernière
    writeDataOnImage(img_path, img_size, detected_face_coordinates, label, loss)