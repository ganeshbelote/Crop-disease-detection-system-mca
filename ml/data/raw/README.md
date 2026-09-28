# Place the real dataset here

This folder is intentionally empty in the repository (the dataset is not
shipped with the project — see the "Dataset Setup" section of the main
README).

Download PlantVillage (or an equivalent crop-disease leaf image dataset)
and arrange it as:

```
ml/data/raw/
├── Tomato_Healthy/
│   ├── image_0001.jpg
│   └── ...
├── Tomato_Early_Blight/
│   └── ...
├── Tomato_Late_Blight/
│   └── ...
├── Potato_Healthy/
│   └── ...
├── Potato_Early_Blight/
│   └── ...
└── Potato_Late_Blight/
    └── ...
```

Folder names become the class names used throughout training, evaluation
and the API response — keep them exactly as above (or update
`ml/src/config.py: DEFAULT_CLASSES` if you use different names).

For a quick, no-download smoke test of the pipeline (NOT real data, see
`ml/src/generate_synthetic_demo_data.py`), you can instead point the
training scripts at `ml/data/demo_synthetic/` with `--data-dir`.
