
# INNER EAR MRI ANALYSIS PROJECT


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import tkinter as tk
from tkinter import filedialog

# IMAGE UPLOAD

root = tk.Tk()

# File window front-la open aaganum
root.attributes('-topmost', True)

file_path = filedialog.askopenfilename(
    title="Select MRI Scan Image",
    filetypes=[("Image Files","*.png *.jpg *.jpeg *.bmp")]
)

root.destroy()

# If no image selected
if file_path == "":
    print("No image selected")
    exit()

# Read image
image = cv2.imread(file_path)

if image is None:
    print("Image not loaded")
    exit()

# CONVERT TO GRAYSCALE

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.title("Grayscale MRI Image")
plt.axis("off")
plt.show()

# FEATURE EXTRACTION

brightness = np.mean(gray)
noise = np.std(gray)

glcm = graycomatrix(gray,
                    distances=[5],
                    angles=[0],
                    levels=256,
                    symmetric=True,
                    normed=True)

contrast = graycoprops(glcm, 'contrast')[0, 0]


h, w = gray.shape
left = gray[:, :w//2]
right = gray[:, w//2:]

symmetry_diff = abs(np.mean(left) - np.mean(right))


# RISK SCORE


risk_score = (noise * 0.5) + (symmetry_diff * 2) + (contrast * 0.01) + (brightness * 0.2)

risk = min(int(risk_score / 3), 100)


# DISEASE DATABASE


if risk < 30:

    disease = "Normal Inner Ear"
    cause = "No abnormality detected"

    symptoms = """
No major symptoms
"""

    stage = "Normal"
    future = "Low risk"
    memory = 90

elif risk < 50:

    disease = "Vestibular Neuritis"
    cause = "Vestibular nerve viral infection"

    symptoms = """
• Sudden dizziness
• Balance difficulty
• Nausea
• Walking instability
"""

    stage = "Mild Stage"
    future = "Possible vertigo episodes"
    memory = 80

elif risk < 75:

    disease = "Meniere's Disease"
    cause = "Fluid imbalance inside inner ear"

    symptoms = """
• Severe vertigo attacks
• Hearing fluctuation
• Ear fullness
• Tinnitus
"""

    stage = "Moderate Stage"
    future = "Hearing loss risk"
    memory = 65

else:

    disease = "Labyrinthitis"
    cause = "Inner ear viral or bacterial infection"

    symptoms = """
• Vertigo
• Hearing loss
• Fever
• Severe balance loss
"""

    stage = "Severe Stage"
    future = "Chronic vertigo and hearing loss"
    memory = 50

# OUTPUT REPORT

print("\n==============================")
print("INNER EAR MRI ANALYSIS REPORT")
print("==============================")

print("\nPredicted Disease:", disease)

print("\nCause:")
print(cause)

print("\nSymptoms:")
print(symptoms)

print("\nDetected Stage:", stage)

print("\nRisk Percentage:", risk,"%")

print("\nMemory Impact Score:", memory)

print("\nFuture Possibility:")
print(future)

print("\nRecommended Tests:")
print("• Audiometry Hearing Test")
print("• Vestibular Balance Test")
print("• ENT Consultation")
print("• Follow-up MRI")

# GRAPH OUTPUT

plt.figure()
plt.bar(["Risk Level"], [risk])
plt.ylim(0,100)
plt.title("Inner Ear Disorder Risk")
plt.show()

plt.figure()
plt.bar(["Memory Score"], [memory])
plt.ylim(0,100)
plt.title("Memory Impact")
plt.show()
