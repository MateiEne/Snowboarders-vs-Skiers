# datele sunt importante => analiza pe date (dimensiunea imaginilor, volum, calitate). Ex: de testat ce se intampla pe dimensiunea imaginilor de 100 x 100

import os

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix



# prima etapa este incarcarea setului de date si preprocesarea imaginilor ca semnale 2D
# setul de date este imparti si el in 3 categorii:
#       training set (70 - 80 %)
#       validation set (15 - 10 %)
#       testing set (15 - 10 %)

# Function to load train images from a folder and assign labels
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g. 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)

                            images.append(img)
                            labels.append(subfolder)

                            # blur
                            blurImg = cv2.GaussianBlur(img, (7, 7), 0)

                            images.append(blurImg)
                            labels.append(subfolder)

                            # rotate
                            # rotatedImg = rotate_image(img, 10)
                            #
                            # images.append(rotatedImg)
                            # labels.append(subfolder)
                            #
                            # rotatedImg = rotate_image(img, -10)
                            #
                            # images.append(rotatedImg)
                            # labels.append(subfolder)

                            # flip
                            flipImg = cv2.flip(img, 0)

                            images.append(flipImg)
                            labels.append(subfolder)

                            print('Labels\n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return images, labels


# Function to load test images from a folder and assign labels
def load_test_images_from_folder(folder, targe_shape=None):
    test_images = []
    test_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape
                        if targe_shape is not None:
                            img = cv2.resize(img, targe_shape)

                        test_images.append(img)
                        test_labels.append(subfolder)
                    else:
                        print(f"Warning: unable to laod {filename}")

    return test_images, test_labels


# Function to plot images grouped by clusters
def plot_cluster_images(images, cluster_assignments, titles):
    n_clusters = len(images)

    plt.Figure(figsize=(15, 5))

    for cluster in range(n_clusters):
        cluster_images = images[cluster]
        cluster_title = f"Fluter {cluster} ({len(cluster_images)} images)"

        for i, image in enumerate(cluster_images):
            plt.subplot(n_clusters, len(cluster_images), i + 1 + cluster * len(cluster_images))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image_idx = np.where(cluster_assignments == cluster)[0][i]  # Find the indices for the current cluster

            plt.title(f"{titles[image_idx]}")
            plt.axis('off')

    plt.show()


# Folder paths
data_folder = './dataset'
test_folder = './test'

data_folder2 = './dataset2'
test_folder2 = './test2'

# Load images and labels from the 'dataset; folder and resize them to (100, 100)
# images, labels = load_train_images_from_folder(data_folder2, target_shape=(200, 200))
images, labels = load_train_images_from_folder(data_folder, target_shape=(200, 200))

# Apply kMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Reshape the images and convert them to grayscale
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]

# Convert the list of 1D arrays to a 2D numpy array
image_data = np.array(image_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)

# Train the k-Means model
kmeans.fit(scaled_data)

# Get cluster assignments
cluster_assignments = kmeans.labels_

# Group images by clusters
cluster_images = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels):
    cluster = cluster_assignments[i]
    cluster_images[cluster].append(images[i])

# Assign titles for images in each cluster
titles = labels

# Plot images grouped by clusters
plot_cluster_images(cluster_images, cluster_assignments, titles)

### TESTING ###
# Load TEST images from the 'test' folder
# test_images, test_labels = load_test_images_from_folder(test_folder2, targe_shape=(200, 200))
test_images, test_labels = load_test_images_from_folder(test_folder, targe_shape=(200, 200))
print('# TEST files:', len(test_images))

# Apply k-Means to TEST images
for i, test_image in enumerate(test_images):
    test_image_data = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).flatten()
    test_image_data = np.array(test_image_data).reshape(1, -1)  # Reshape for a single sample

    scaled_test_data = scaler.transform(test_image_data)

    test_predicion = kmeans.predict(scaled_test_data)  # Predict classes

    if test_predicion[0] == 1:
        print(f"Test Image {i + 1} - ==SNOWBOARDER==")
    else:
        print(f"Test Image {i + 1} - ==SCHIOR==")

    # Visualize the test image with its predicted cluster
    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

    if test_predicion[0] == 1:
        plt.title(f"Test Image {i + 1} - ==SNOWBOARDER==")
    else:
        plt.title(f"Test Image {i + 1} - ==SCHIOR==")

    plt.axis('off')
    plt.show()

# Crearea unui fișier CSV și scrierea rezultatelor
csv_file_path = 'rezultate_test.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label'])

    for i, test_image in enumerate(test_images):
        test_image_data = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).flatten()
        test_image_data = np.array(test_image_data).reshape(1, -1)  # Reshape for a single sample

        scaled_test_data = scaler.transform(test_image_data)

        test_prediction = kmeans.predict(scaled_test_data)  # Predict classes

        true_label = 'SNOWBOARDER' if 'snowboarder' in test_labels[i].lower() else 'SCHIOR'
        predicted_label = 'SNOWBOARDER' if test_prediction[0] == 1 else 'SCHIOR'

        # Adăugare rând în fișierul CSV
        writer.writerow([true_label, predicted_label])

# Afișarea calei fișierului CSV
print(f'Rezultatele au fost salvate în: {csv_file_path}')

# Calcularea acurateței
true_labels_numeric = [1 if 'snowboarder' in label.lower() else 0 for label in test_labels]
accuracy = accuracy_score(true_labels_numeric, cluster_assignments)

# Calcularea preciziei pentru cluster-ul cu snowboarderi (1)
precision_snowboarder = precision_score(true_labels_numeric, cluster_assignments, pos_label=1)

# Calcularea preciziei pentru cluster-ul cu schiori (0)
precision_schior = precision_score(true_labels_numeric, cluster_assignments, pos_label=0)

# Afișarea rezultatelor
print(f'Acuratețe: {accuracy * 100:.2f}%')
print(f'Precizie (SNOWBOARDER): {precision_snowboarder * 100:.2f}%')
print(f'Precizie (SCHIOR): {precision_schior * 100:.2f}%')

# Calcularea matricei de confuzie
confusion_mat = confusion_matrix(true_labels_numeric, cluster_assignments)

# Afișarea matricei de confuzie
print("Matricea de confuzie:")
print(confusion_mat)
