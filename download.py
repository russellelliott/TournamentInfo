import gdown

#https://drive.google.com/file/d/13Cwj92E_hf7DQaT4D8PxhzR6FXxJB6Gw/view

# File ID from the URL
file_id = "13Cwj92E_hf7DQaT4D8PxhzR6FXxJB6Gw"
url = f"https://drive.google.com/uc?id={file_id}"

# Output filename
output = "CCLSpring2025.pdf"

# Download the file
gdown.download(url, output, quiet=False) #instead of downloading it locally, should be uplaoded to firebase
