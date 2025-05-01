import os
import duckdb
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import time

# 1) Connect (in-memory or to a file: duckdb.connect("mydata.duckdb"))
con = duckdb.connect()

# 2) Discover all image files and insert into a DuckDB table
IMG_DIR = "/Users/baveethora/Desktop/ Velox/idimage"   # ← notice the space before "Velox"


# 2) Once you confirm the real folder name, update IMG_DIR accordingly.
#    For example, if Desktop contains "Velox" (no space), then:
#    IMG_DIR = "/Users/baveethora/Desktop/Velox/idimage"

# 3) Walk it again:
paths = []
for root, dirs, files in os.walk(IMG_DIR):
       # you should see your "fraud" and "nonfraud" dirs here
    label = os.path.basename(root)    # will be "fraud", "nonfraud", or "idimage"
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            full = os.path.join(root, f)
            paths.append((full, label))

# 3) Create & populate the table
# 1) Create a two-column table
con.execute("""
  CREATE TABLE IF NOT EXISTS image_table (
    img_path     VARCHAR,
    true_label   VARCHAR
  )
""")

# 2) Insert with two placeholders
con.executemany(
    "INSERT INTO image_table VALUES(?, ?)",
    paths   # paths is [(full_path, label), ...]
)


# 4) (Rest of your UDF registration & inference…)
sess = ort.InferenceSession("smallcnn.onnx")

preproc = transforms.Compose([
    transforms.Resize((128, 128)),           # resize to match your training
    transforms.ToTensor(),                   # convert PIL→torch.Tensor
    transforms.Normalize(                    # same normalization you used in training
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])

def predict_fraud(path: str) -> float:
    """
    Given an image file path, returns P(fraud) in [0,1].
    """
    img = Image.open(path).convert("RGB")
    x = preproc(img).unsqueeze(0).numpy().astype(np.float32)  # (1,3,128,128)
    logit = sess.run(None, {"image": x})[0][0, 0]             # raw output
    prob = 1 / (1 + np.exp(-logit))                          # sigmoid
    return float(prob)

# Explicit SQL types
# Let DuckDB infer
con.create_function(
    name="predict_fraud",
    function=predict_fraud,
    parameters=["VARCHAR"],   # list of SQL types for each arg
    return_type="DOUBLE"      # SQL return type
)


# Now your SQL will resolve predict_fraud(...) as a UDF:
tic = time.perf_counter()

df = con.execute("""
WITH scored AS (
  SELECT
    img_path,
    predict_fraud(img_path) AS fraud_prob
  FROM image_table
)
SELECT
  img_path,
  fraud_prob,
  CASE
    WHEN fraud_prob > 0.5 THEN 'not_fraud'
    ELSE 'fraud'
  END AS pred_label
FROM scored
WHERE fraud_prob > 0.95           -- only keep high‐risk
ORDER BY fraud_prob DESC;        -- rank by probability
""").fetchdf()

toc = time.perf_counter()

print(f"Execution time for the query : {toc - tic:0.4f} seconds")

print(df)