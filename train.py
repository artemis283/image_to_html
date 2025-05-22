import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikramtiwari/pix2code")

print("Path to dataset files:", path)