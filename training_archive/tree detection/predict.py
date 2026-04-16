from deepforest import main
import matplotlib.pyplot as plt
import cv2

# Load model
model = main.deepforest()
model.load_model()

image_path = "./dataset/images/bishop_2020_7.tif"

# Read image
image = cv2.imread(image_path)

# Predict
predictions = model.predict_image(path=image_path)

# Filter low confidence
# Predict
predictions = model.predict_image(path=image_path)

# Threshold
predictions = predictions[predictions["score"] > 0.25]

# Size filter
predictions = predictions[
    (predictions["xmax"] - predictions["xmin"] > 10) &
    (predictions["ymax"] - predictions["ymin"] > 10)
]

print(predictions)

tree_count = len(predictions)
print(f"Total Trees Detected: {tree_count}")

# Draw bounding boxes manually
for _, row in predictions.iterrows():
    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0))

# Convert BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.putText(
    image,
    f"Trees: {tree_count}",
    (10, 30),
    cv2.FONT_HERSHEY_DUPLEX,
    0.5,
    (0, 255, 0),
    2
)

# Show image
plt.imshow(image)
plt.axis("off")
plt.show()