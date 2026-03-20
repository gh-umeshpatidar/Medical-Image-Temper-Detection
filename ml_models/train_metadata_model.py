import os
import numpy as np
import lightgbm as lgb
import joblib
import pydicom
from tqdm.asyncio import tqdm
from app.config.settings import METADATA_MODEL, label_map
from app.services.dicom_metadata import DATASET_PATH, extract_metadata

def create_npy_files():
    # Traverse dataset
    for cls in os.listdir(DATASET_PATH):
    
        cls_path = os.path.join(DATASET_PATH, cls)
    
        for scan in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):
    
            scan_path = os.path.join(cls_path, scan)
    
            for file in os.listdir(scan_path):
    
                if file.endswith(".dcm"):
                    path = os.path.join(scan_path, file)
    
                    try:
                        ds = pydicom.dcmread(path)
    
                        features = extract_metadata(ds)
                        
                        if features is not None:
                            X.append(features)
                            Y.append(label_map[cls])
                            # print(features, "->", label_map[cls])
    
                    except Exception as e:
                        print(f"⚠️  Failed to read {path}: {e}")
                        continue

X = joblib.load("metadata_features.npy") if os.path.exists("metadata_features.npy") else None
Y = joblib.load("labels.npy") if os.path.exists("labels.npy") else None

if X is None or Y is None:
    print("⏳ Extracting metadata features and creating .npy files...")
    create_npy_files()

# Convert to numpy
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)

print("Feature shape:", X.shape)
print("Label shape:", Y.shape)


# Save files
np.save("metadata_features.npy", X)
np.save("labels.npy", Y)

print("✅ metadata_features.npy created")
print("✅ labels.npy created")









X = np.load("metadata_features.npy")
Y = np.load("labels.npy")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64
)


model.fit(X, Y)
joblib.dump(model, METADATA_MODEL)

print("Model trained & saved.")
