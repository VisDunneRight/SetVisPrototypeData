import os

# List of folders to modify
folders = ["ground_truth", "MultiSpectral", "Thermal", "VideoColor"]

# Iterate through each folder
for folder in folders:
    # Get a list of all .txt files in the folder
    txt_files = [f for f in os.listdir(f"droneProjectedData/{folder}") if f.endswith(".txt")]
    
    # Iterate through each .txt file
    for txt_file in txt_files:
        # Open the file and read its contents
        with open(f"droneProjectedData/{folder}/{txt_file}", "r") as f:
            contents = f.readlines()
        
        # Iterate through each line and modify the x and y values
        new_contents = []
        for line in contents:
            parts = line.split()
            x_min = float(parts[1]) / 1166
            y_min = float(parts[2]) / 744
            x_max = float(parts[3]) / 1166
            y_max = float(parts[4]) / 744
            if len(parts) == 5:
                new_line = f"{parts[0]} {x_min} {y_min} {x_max} {y_max}\n"
            else:
                confidence = parts[5]
                new_line = f"{parts[0]} {x_min} {y_min} {x_max} {y_max} {confidence}\n"
            new_contents.append(new_line)
        
        # Write the modified contents back to the same file
        with open(f"droneProjectedData/{folder}/{txt_file}", "w") as f:
            f.writelines(new_contents)
