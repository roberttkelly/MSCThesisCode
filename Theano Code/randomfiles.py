import random
import shutil

total_files = 100
sample_size = 5.45
orig_directory = "" # Add image Directory
sample_directory = "/"

files = []
for i in range(1, total_files + 1):
    files.append((str(i) + ".tif"))

# Create files to test with. Should be commented out
# for file in files:
#     open(orig_directory + file[0], 'w+')
#     open(orig_directory + file[1], 'w+')

# Select random sample
selected_files = random.sample(files, sample_size)

# Move selected files to sample directory
for file in selected_files:
    shutil.move(orig_directory + file[0], sample_directory + file[0])
    shutil.move(orig_directory + file[1], sample_directory + file[1])