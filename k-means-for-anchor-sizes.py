import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def analyze_anchors(label_dir, img_width=800, img_height=800, n_clusters=5):

    widths = []
    heights = []


    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"{len(label_files)} etiket dosyası taranıyor...")

    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # YOLO format: class x_center y_center width height (normalize)
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                widths.append(w)
                heights.append(h)

    X = np.array(list(zip(widths, heights)))

    # K-Means Kümeleme
    kmeans_size = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_size.fit(X)
    cluster_centers = kmeans_size.cluster_centers_
    
    cluster_centers = cluster_centers[np.argsort(cluster_centers[:, 0] * cluster_centers[:, 1])]

    # Oran Analizi 
    ratios = np.array(widths) / np.array(heights)
    ratios = ratios[(ratios > 0.2) & (ratios < 5.0)]
    
    kmeans_ratio = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_ratio.fit(ratios.reshape(-1, 1))
    best_ratios = np.sort(kmeans_ratio.cluster_centers_.flatten())

    print("\n" + "="*40)
    print("ANALİZ SONUÇLARI")
    print("="*40)
    print(f"Önerilen Anchor Sizes (px):")
    for i, center in enumerate(cluster_centers):
        # Alanın karekökünü alarak 'size' parametresine dönüştürüyoruz
        side = np.sqrt(center[0] * center[1])
        print(f" Cluster {i+1}: {int(side)}px (W:{int(center[0])}, H:{int(center[1])})")

    print(f"\nÖnerilen Aspect Ratios (W/H):")
    print(f" {best_ratios.round(2)}")
    print("="*40)

    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.1, c='gray', label='Gerçek Kutular')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
    plt.title("Çukur Boyut Dağılımı ve K-Means Merkezleri")
    plt.xlabel("Genişlik (px)")
    plt.ylabel("Yükseklik (px)")
    plt.legend()
    plt.show()

analyze_anchors('/Users/mustafa/Library/Mobile Documents/com~apple~CloudDocs/Computer_Science/09-Makalelerim/6-ADAS/Dataset_Mendeley/labels/train', img_width=800, img_height=800)