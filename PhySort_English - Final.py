import sys

REQUIRED_PYTHON_VERSION = (3, 6)
if sys.version_info < REQUIRED_PYTHON_VERSION:
    sys.exit(f"Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} or higher is required. Current version: {sys.version_info.major}.{sys.version_info.minor}")

try:
    import os
    import shutil
    from datetime import datetime
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, font
    from PIL import Image, ImageTk
    import cv2
    import face_recognition
    import numpy as np
    import threading
    from PIL.ExifTags import TAGS
    import logging
except ImportError as e:
    sys.exit(f"Error importing a library, or library not installed: {e}")

# Parameters
BLUR_THRESHOLD_DEFAULT = 100.0 # Default value for blur check
BRIGHTNESS_THRESHOLD_DEFAULT = 200 # Default value for brightness check
DARKNESS_THRESHOLD_DEFAULT = 50 # Default value for brightness check
DISABLED_COLOR = 'gray' # Color for disabled scale
ENABLED_COLOR = 'blue' # Color for enabled scale
FILE_TYPES = [("Image files", "*.png;*.jpg;*.jpeg")] # Supported file types
FACE_RECON_SIZE = (1280, 1024) # Size of the face recognition window
NC_FOLDER = "NonConforming" # Folder for non-conforming images
LOG_FILE = "sort_photos_log.txt" # Log file
MAX_WIDTH = 5000 # Max slider value, width
MAX_HIGH = 5000 # Max slider value, height
MAX_SIZE = 10000 # Max slider value, file size

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# Logging function
def log_message(message):
    print(message)
    logging.info(message)
    
# Configure scale
def configure_scale(scale, state, troughcolor):
    scale.config(state=state, troughcolor=troughcolor)
    log_message(f"Scales configured: {scale} to state {state}")

# Blur check
def is_blurry(image, threshold=BLUR_THRESHOLD_DEFAULT):
    img_cv = convert_to_grayscale(image)
    laplacian_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
    return laplacian_var < threshold

# Blur slider
def on_check_blur(*args):
    if check_blur_var.get():
        configure_scale(blur_threshold_scale, tk.NORMAL, ENABLED_COLOR)
        if not selected_image_path.get():
            select_image()
    else:
        configure_scale(blur_threshold_scale, tk.DISABLED, DISABLED_COLOR)
    update_example_image()
    log_message(f"Blur check {'enabled' if check_blur_var.get() else 'disabled'}.")

# Brightness check
def is_bad_lighting(image, middle_threshold=100, range_value=50):
    img_cv = convert_to_grayscale(image)
    mean_brightness = np.mean(img_cv)
    brightness_threshold = middle_threshold + range_value
    darkness_threshold = middle_threshold - range_value
    return mean_brightness > brightness_threshold or mean_brightness < darkness_threshold

# Brightness slider
def on_check_brightness(*args):
    if check_brightness_var.get():
        configure_scale(middle_threshold_scale, tk.NORMAL, ENABLED_COLOR)
        if not selected_image_path.get():
            select_image()
    else:
        configure_scale(middle_threshold_scale, tk.DISABLED, DISABLED_COLOR)
    update_example_image()
    log_message(f"Brightness check {'enabled' if check_brightness_var.get() else 'disabled'}.")

# Screenshot check
def is_screenshot(file_path, exif_data):
    filename = os.path.basename(file_path).lower()
    software = exif_data.get('Software', '').lower()
    return 'screenshot' in filename or 'screenshot' in software or 'screen capture' in software

# Checkbox screenshot
def on_check_screenshot(*args):
    global photo_path
    if check_screenshot_var.get():
        if photo_path:
            photo_path = ""
            log_message("Face recognition disabled because screenshot detection was enabled.")
    example_image_label.config(image=None)
    example_image_label.image = None
    log_message(f"Screenshot detection {'enabled' if check_screenshot_var.get() else 'disabled'}.")

# File format check
def is_supported_format(filename):
    extensions = []
    for description, ext in FILE_TYPES:
        extensions.extend(ext.split(';'))
    supported_extensions = tuple(ext.replace('*', '') for ext in extensions)
    return filename.lower().endswith(supported_extensions)

# EXIF data Creation Date
def get_exif_creation_date(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if (exif_data):
            for tag, value in exif_data.items():
                if TAGS.get(tag, tag) == 'DateTimeOriginal':
                    return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except (IOError, ValueError) as e:
        log_message(f"Error reading creation date from {file_path}: {e}")
    return None

# Read EXIF data
def get_exif_data(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if exif_data is not None:
            exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
            return exif
    except Exception as e:
        log_message(f"Error reading EXIF data from {file_path}: {e}")
    return {}

# Select folder
def select_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)
        log_message(f"Folder selected: {folder}")

# Select example image for blur and brightness check
def select_image():
    global photo_path
    photo_path = ''  # Delete the previous photo path if it exists
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_image_path.set(file_path)
        update_example_image()
        log_message(f"Example image: {file_path}")

# Source file Processing
def process_file(file_path, min_width, min_height, min_file_size, blur_threshold, check_blur, brightness_threshold, darkness_threshold, copy_non_conforming, non_conforming_folder, check_screenshot_var, check_brightness, destination_folder=None, include_subfolder_name=False, root_dir=None):
    filename = os.path.basename(file_path)
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if (min_width > 0 and width < min_width) or (min_height > 0 and height < min_height):
                log_message(f"Skipping file {filename}: Image dimensions too small ({width}x{height})")
                if copy_non_conforming:
                    if not os.path.exists(non_conforming_folder):
                        os.makedirs(non_conforming_folder)
                        log_message(f"Create NC folder: {non_conforming_folder}")
                    shutil.copy(file_path, non_conforming_folder)
                return False
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    except Exception as e:
        log_message(f"Error opening image {file_path}: {e}")
        if copy_non_conforming:
            if not os.path.exists(non_conforming_folder):
                os.makedirs(non_conforming_folder)
                log_message(f"Create NC folder: {non_conforming_folder}")
            shutil.copy(file_path, non_conforming_folder)
        return False

    if min_file_size > 0 and os.path.getsize(file_path) < min_file_size:
        log_message(f"Skipping file {filename}: File size too small ({os.path.getsize(file_path)} Bytes)")
        if copy_non_conforming:
            if not os.path.exists(non_conforming_folder):
                os.makedirs(non_conforming_folder)
                log_message(f"Create NC folder: {non_conforming_folder}")
            shutil.copy(file_path, non_conforming_folder)
        return False

    file_stat = os.stat(file_path)
    img_cv = cv2.imread(file_path)

    # Check if Screenshot is enabled
    if check_screenshot_var.get():
        exif_data = get_exif_data(file_path)
        if not is_screenshot(file_path, exif_data):
            log_message(f"Skipping file {filename}: Not a screenshot")
            if copy_non_conforming:
                if not os.path.exists(non_conforming_folder):
                    os.makedirs(non_conforming_folder)
                    log_message(f"Create NC folder: {non_conforming_folder}")
                shutil.copy(file_path, non_conforming_folder)
            return False

    # Check if the blur check is enabled
    if check_blur:
        img_cv = np.uint8(img_cv)
        if is_blurry(img_cv, blur_threshold):
            log_message(f"Skipping file {filename}: Image is blurry")
            if copy_non_conforming:
                if not os.path.exists(non_conforming_folder):
                    os.makedirs(non_conforming_folder)
                    log_message(f"Create NC folder: {non_conforming_folder}")
                shutil.copy(file_path, non_conforming_folder)
            return False

    # Check if the brightness check is enabled
    if check_brightness:
        if brightness_threshold > 0 or darkness_threshold > 0:
            if is_bad_lighting(img_cv, brightness_threshold, darkness_threshold):
                log_message(f"Skipping file {filename}: Poor lighting")
                if copy_non_conforming:
                    if not os.path.exists(non_conforming_folder):
                        os.makedirs(non_conforming_folder)
                        log_message(f"Create NC folder: {non_conforming_folder}")
                    shutil.copy(file_path, non_conforming_folder)
                return False

    # Create destination folder
    if destination_folder:
        creation_time = get_exif_creation_date(file_path)
        if creation_time is None:
            try:
                # Use the st_ctime timestamp
                creation_time = datetime.fromtimestamp(file_stat.st_ctime)
            except AttributeError:
                # Fallback to st_mtime if st_ctime is not available
                creation_time = datetime.fromtimestamp(file_stat.st_mtime)

        year_folder = creation_time.strftime('%Y')
        month_folder = creation_time.strftime('%m')
        
        if include_subfolder_name and root_dir:
            source_subfolder_name = os.path.basename(root_dir)
            date_folder_path = os.path.join(destination_folder, year_folder, month_folder, source_subfolder_name)
        else:
            date_folder_path = os.path.join(destination_folder, year_folder, month_folder)

        if not os.path.exists(date_folder_path):
            os.makedirs(date_folder_path)
            log_message(f"Destination folder created: {date_folder_path}")

        destination_path = os.path.join(date_folder_path, filename)
        shutil.copy(file_path, destination_path)
        log_message(f"Image {filename} copied.")

    return True

# Copy files to destination folder
def sort_photos_by_date(source_folder, destination_folder, include_subfolder_name, progress_var, progress_label, root, min_width, min_height, min_file_size, blur_threshold, check_blur, brightness_threshold, darkness_threshold, copy_non_conforming, stop_event, check_screenshot_var, check_brightness=False):
    photo_count = 0
    folder_count = 0
    processed_files = 0
    non_conforming_folder = os.path.join(destination_folder, NC_FOLDER)
    
    # Calculate the total number of files in the source folder
    total_files = sum([len(files) for _, _, files in os.walk(source_folder)])
    
    for root_dir, _, files in os.walk(source_folder):
        for filename in files:
            if stop_event.is_set():
                return photo_count, folder_count

            if not is_supported_format(filename):
                log_message(f"Skip file {filename}: Unsupported file format")
                continue

            file_path = os.path.join(root_dir, filename)
            processed = process_file(file_path, min_width, min_height, min_file_size, blur_threshold, check_blur, brightness_threshold, darkness_threshold, copy_non_conforming, non_conforming_folder, check_screenshot_var, check_brightness, destination_folder, include_subfolder_name, root_dir)
            if processed:
                photo_count += 1

            processed_files += 1
            progress_var.set((processed_files / total_files) * 100)
            if progress_label.winfo_exists():  # Check if the progress_label widget still exists
                progress_label.config(text=f"Processed files: {processed_files}/{total_files}")
            root.update_idletasks()
    return photo_count, folder_count

# Process update_example_image
def preprocess_image(file_path):
    log_message(f"Starting preprocessing of image: {file_path}")
    img_cv = cv2.imread(file_path)
    
    # Blur check
    if check_blur_var.get():
        img_cv = np.uint8(img_cv)
        blur_amount = blur_threshold_scale.get()
        if blur_amount > 0:
            ksize = (blur_amount // 2) * 2 + 1  # Kernel size must be odd
            img_cv = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)
            img_cv = cv2.putText(img_cv, 'Blurry', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
            log_message(f"Blur applied with threshold: {blur_amount}")
    
    # Brightness check
    if check_brightness_var.get():
        mean_brightness = np.mean(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)) 
        darkness_threshold = middle_threshold_scale.get() - 50
        brightness_threshold = middle_threshold_scale.get() + 50
        if mean_brightness < darkness_threshold:
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1.5, beta=50)
        elif mean_brightness > brightness_threshold:
            img_cv = cv2.convertScaleAbs(img_cv, alpha=0.5, beta=-50)
        
        # Check if the text "Bad Lighting" is already present (does not work correctly)
        if is_bad_lighting(img_cv, middle_threshold_scale.get(), 35):
            if 'Bad Lighting' not in img_cv:
                img_cv = cv2.putText(img_cv, 'Bad Lighting', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                log_message(f"Brightness adjustment applied with middle threshold: {middle_threshold_scale.get()}")
    
    return img_cv

# Update image in frame
def update_example_image():
    file_path = selected_image_path.get()
    if file_path:
        img_cv = preprocess_image(file_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) 
        
        max_width, max_height = 800, 600  # Size of the image in the window
        height, width, _ = img_cv.shape
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img_cv = cv2.resize(img_cv, new_size)

        img_pil = Image.fromarray(img_cv)
        img_tk = ImageTk.PhotoImage(img_pil)
        example_image_label.config(image=img_tk)
        example_image_label.image = img_tk
        log_message(f"Update example image: {file_path}")

# Start sorting, except face recognition
def start_sorting():
    source_folder = source_folder_entry.get()
    destination_folder = destination_folder_entry.get()

    # Check if source and destination folders are selected
    if not source_folder:
        messagebox.showerror("Error", "Please select a source folder.")
        log_message("Error: No source folder selected.")
        return
    if not destination_folder:
        messagebox.showerror("Error", "Please select a destination folder.")
        log_message("Error: No destination folder selected.")
        return
    
    if photo_path:
        start_search()
        log_message("Start face recognition.") 
        return
    
    include_subfolder_name = include_subfolder_name_var.get()
    min_width = min_width_scale.get()
    min_height = min_height_scale.get()
    min_file_size = min_file_size_scale.get() * 1024  # Convert from KB to Bytes
    blur_threshold = blur_threshold_scale.get()
    check_blur = check_blur_var.get()
    brightness_threshold = middle_threshold_scale.get() + 50
    darkness_threshold = middle_threshold_scale.get() - 50
    copy_non_conforming = copy_non_conforming_var.get()
    check_screenshot = check_screenshot_var.get()
    check_brightness = check_brightness_var.get()  

    log_message(f"Starting sorting: Source={source_folder}, Destination={destination_folder}")

    # Create a new window for the progress bar
    progress_window = tk.Toplevel(root)
    progress_window.title("Progress")
    progress_window.geometry("400x150")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
    progress_bar.pack(pady=20, padx=20, fill=tk.X)

    progress_label = tk.Label(progress_window, text="Processed files: 0")
    progress_label.pack(pady=5)

    stop_event = threading.Event()

    def stop_sorting():
        stop_event.set()
        progress_window.destroy()
        log_message("Sorting stopped.")

    stop_button = tk.Button(progress_window, text="Stop", command=stop_sorting)
    stop_button.pack(pady=5)

    def run_sorting():
        photo_count, folder_count = sort_photos_by_date(
            source_folder, destination_folder, include_subfolder_name, progress_var, progress_label, root,
            min_width, min_height, min_file_size, blur_threshold, check_blur, brightness_threshold, darkness_threshold, copy_non_conforming, stop_event, check_screenshot_var, check_brightness  
        )
        if not stop_event.is_set():
            try:
                if progress_label.winfo_exists():  # Check if the progress_label widget still exists
                    messagebox.showinfo("Done", f"Sorting completed! {photo_count} photos sorted into {folder_count} folders.\nDestination folder: {destination_folder}")
                    log_message(f"Sorting completed: {photo_count} photos sorted into {folder_count} folders.\nDestination folder: {destination_folder}")
            except tk.TclError:
                pass  
        progress_window.destroy()

    threading.Thread(target=run_sorting).start()

# From here, face recognition
def detect_faces(image_path):
    image = cv2.imread(image_path)
    img_cv = convert_to_grayscale(image)
    face_locations = face_recognition.face_locations(img_cv)
    if not face_locations:
        log_message("No faces detected.")
        messagebox.showerror("Error", "No faces detected.")
    return image, face_locations

# Mark faces
def mark_faces(image, face_locations, selected_face_location=None):
    log_message("Mark faces.")
    if not face_locations:
        log_message("No faces found to mark.")
        messagebox.showerror("Error", "No faces found to mark.")
        return None

    for (top, right, bottom, left) in face_locations:
        if (top, right, bottom, left) == selected_face_location:
            color = (0, 0, 255)
            log_message(f"Mark face at position ({left}, {top}, {right}, {bottom}) in red.")
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    marked_image_path = "target_face.jpg"
    
    # Try to save the image and log an error message if it doesn't work
    try:
        if cv2.imwrite(marked_image_path, image):
            log_message(f"Image successfully saved at: {marked_image_path}")
        else:
            log_message(f"Error saving image: {marked_image_path}")
            return None
    except Exception as e:
        log_message(f"Error saving image: {e}")
        return None
    
    log_message("Image successfully saved. Loading image into frame.")
    
    # Load the image into the frame
    try:
        img_pil = Image.open(marked_image_path)
        
        max_size = (800, 600)
        img_pil.thumbnail(max_size, Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        example_image_label.config(image=img_tk)
        example_image_label.image = img_tk
        log_message("Image successfully loaded into frame.")
    except Exception as e:
        log_message(f"Error loading image into frame: {e}")
        return None
    
    return marked_image_path

# Check if face was clicked
def click_event(event, x, y, flags, param):
    global selected_face_location
    if event == cv2.EVENT_LBUTTONDOWN:
        for (top, right, bottom, left) in param:
            if left < x < right and top < y < bottom:
                selected_face_location = (top, right, bottom, left)
                log_message(f"Face at position ({left}, {top}, {right}, {bottom}) clicked and selected.")
                cv2.destroyAllWindows()
                break

# Mark face on click
def select_face(image, face_locations):
    global selected_face_location 
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Display the image in an extra window
    cv2.namedWindow("Mark faces", cv2.WINDOW_NORMAL)
    cv2.imshow("Mark faces", image)
    cv2.resizeWindow("Mark faces", FACE_RECON_SIZE) 
    cv2.setMouseCallback("Mark faces", click_event, face_locations)
    cv2.waitKey(0)
    
    # Mark selected face red
    if selected_face_location:
        log_message(f"Mark selected face at position {selected_face_location} in red.")
        mark_faces(image, face_locations, selected_face_location)
    
    return selected_face_location

# Copy face recon photos
def search_and_copy_photos(target_face_encoding, source_dir, dest_dir, progress_var, progress_label, root, stop_event, min_width, min_height, min_file_size, copy_non_conforming, non_conforming_folder, include_subfolder_name):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        log_message(f"Destination folder created: {dest_dir}")
    
    if copy_non_conforming and not os.path.exists(non_conforming_folder):
        os.makedirs(non_conforming_folder)
        log_message(f"Non-conforming folder created: {non_conforming_folder}")
    
    total_files = sum([len(files) for _, _, files in os.walk(source_dir)])
    processed_files = 0
    photo_count = 0
    folder_count = 0

    for root_dir, _, files in os.walk(source_dir):
        for file in files:
            if stop_event.is_set():
                return photo_count, folder_count
            if is_supported_format(file):
                file_path = os.path.join(root_dir, file)
                filename = os.path.basename(file_path)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if (min_width > 0 and width < min_width) or (min_height > 0 and height < min_height):
                            log_message(f"Skipping file {filename}: Image dimensions too small ({width}x{height})")
                            if copy_non_conforming:
                                shutil.copy(file_path, non_conforming_folder)
                            continue
                        img_cv = np.array(img)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    log_message(f"Error opening image {file_path}: {e}")
                    if copy_non_conforming:
                        shutil.copy(file_path, non_conforming_folder)
                    continue

                if min_file_size > 0 and os.path.getsize(file_path) < min_file_size:
                    log_message(f"Skipping file {filename}: File size too small ({os.path.getsize(file_path)} Bytes)")
                    if copy_non_conforming:
                        shutil.copy(file_path, non_conforming_folder)
                    continue

                file_stat = os.stat(file_path)
                img_cv = cv2.imread(file_path)

                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image)
                
                for face_encoding in face_encodings:
                    match = face_recognition.compare_faces([target_face_encoding], face_encoding)
                    if match[0]:
                        # Create destination folder                     
                        creation_time = get_exif_creation_date(file_path)
                        if creation_time is None:
                            try:
                                # Use the st_ctime timestamp                              
                                creation_time = datetime.fromtimestamp(file_stat.st_ctime)
                            except AttributeError:
                                # Fallback to st_mtime if st_ctime is not available            
                                creation_time = datetime.fromtimestamp(file_stat.st_mtime)

                        year_folder = creation_time.strftime('%Y')
                        month_folder = creation_time.strftime('%m')
                        if include_subfolder_name and root_dir:
                            source_subfolder_name = os.path.basename(root_dir)
                            date_folder_path = os.path.join(dest_dir, year_folder, month_folder, source_subfolder_name)
                        else:
                            date_folder_path = os.path.join(dest_dir, year_folder, month_folder)

                        if not os.path.exists(date_folder_path):
                            os.makedirs(date_folder_path)
                            folder_count += 1
                        log_message(f"Destination folder created: {date_folder_path}")

                        destination_path = os.path.join(date_folder_path, filename)
                        shutil.copy(file_path, destination_path)
                        log_message(f"Image {filename} copied.")
                        photo_count += 1
                    else:
                        if copy_non_conforming:
                            shutil.copy(file_path, non_conforming_folder)
                            log_message(f"Non-conforming image copied: {file_path}")
            
            # Processing window
            processed_files += 1
            progress = (processed_files / total_files) * 100
            if root.winfo_exists():  # Check if the root widget still exists
                root.after(0, lambda p=progress: progress_var.set(p))
                root.after(0, lambda pf=processed_files, tf=total_files: progress_label.config(text=f"Processed files: {pf}/{tf}"))
            root.update_idletasks()

    return photo_count, folder_count

# Start search for photos with selected face
def start_search():
    global selected_face_location, photo_path 
    
    source_dir = source_folder_entry.get()
    dest_dir = destination_folder_entry.get()
    
    if not photo_path or not source_dir or not dest_dir:
        messagebox.showerror("Error", "Please fill in all fields.")
        return
    
    if selected_face_location is None:
        messagebox.showerror("Error", "No face selected.")
        return
     
    image = face_recognition.load_image_file(photo_path)
    target_face_encoding = face_recognition.face_encodings(image, [selected_face_location])[0]

    # Create a new window for the progress bar
    progress_window = tk.Toplevel(root)
    progress_window.title("Progress")
    progress_window.geometry("400x150")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
    progress_bar.pack(pady=20, padx=20, fill=tk.X)

    progress_label = tk.Label(progress_window, text="Processed files: 0")
    progress_label.pack(pady=5)

    stop_event = threading.Event()
    photo_path = ""

    def stop_search():
        stop_event.set()
        progress_window.destroy()
        log_message("Search stopped.")

    stop_button = tk.Button(progress_window, text="Stop", command=stop_search)
    stop_button.pack(pady=5)

    def run_search():
        min_width = min_width_scale.get()
        min_height = min_height_scale.get()
        min_file_size = min_file_size_scale.get() * 1024  # Convert from KB to Bytes
        copy_non_conforming = copy_non_conforming_var.get()
        non_conforming_folder = os.path.join(dest_dir, NC_FOLDER)
        include_subfolder_name = include_subfolder_name_var.get()

        photo_count, folder_count = search_and_copy_photos(
            target_face_encoding, source_dir, dest_dir, progress_var, progress_label, root, stop_event, 
            min_width, min_height, min_file_size, copy_non_conforming, non_conforming_folder, include_subfolder_name
        )
        
        if not stop_event.is_set():
            try:
                if progress_label.winfo_exists():  # Check if the progress_label widget still exists
                    messagebox.showinfo("Done", f"Sorting completed! {photo_count} photos sorted into {folder_count} folders.\nDestination folder: {dest_dir}")
                    log_message(f"Sorting completed: {photo_count} photos sorted into {folder_count} folders.\nDestination folder: {dest_dir}")
            except tk.TclError:
                pass 
        progress_window.destroy()

    threading.Thread(target=run_search).start()

# Start face recon
def target_face():
    log_message("Start face recognition.")
    global selected_face_location, photo_path
    photo_path = filedialog.askopenfilename(title="Select file", filetypes=FILE_TYPES)
     
    if photo_path:
        image, face_locations = detect_faces(photo_path)
        selected_face_location = select_face(image, face_locations)
        if selected_face_location is None:
            messagebox.showerror("Error", "No face selected.")
        else:
            # Call the function mark_faces
            marked_image_path = mark_faces(image, face_locations, selected_face_location)
            if marked_image_path:
                log_message(f"Faces successfully marked and image saved at: {marked_image_path}")
            else:
                log_message("Error marking faces and saving image.")

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Help window
def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.geometry("1024x850")
    help_window.configure(bg="white")

    # Define fonts
    bold_font = font.Font(family="Helvetica", size=10, weight="bold")
    normal_font = font.Font(family="Helvetica", size=9)

    help_text = """
1. Starting the program
- Make sure all required libraries are installed (tkinter, PIL, cv2, face_recognition, numpy, threading, logging).
- Run the Python script PhySort.py.

2. Selecting the source and destination folders
- Click the "Browse" button next to the "Source Folder" field to select the folder containing the photos to be sorted.
- Click the "Browse" button next to the "Destination Folder" field to select the folder where the sorted photos will be copied.

3. Sorting options
- Include subfolder names: Enable this option if you want to keep the names of the subfolders in the source folder in the destination folders.
- Copy non-conforming images separately: Enable this option if you want to copy images that do not meet the specified requirements to a separate folder.

4. Setting image requirements
- Minimum width: Set the minimum width of the images to be copied.
- Minimum height: Set the minimum height of the images to be copied.
- Minimum file size (KB): Set the minimum file size of the images to be copied.

5. Blur check (BETA)
- Enable the "Check blur" option to check images for blur.
- Set the blur threshold using the slider.

6. Brightness check (BETA)
- Enable the "Check brightness" option to check images for poor lighting.
- Set the middle brightness threshold using the slider.

7. Screenshot detection
- Enable the "Copy only screenshots" option to copy only screenshots.

8. Select example image (BETA)
- Click the "Select example image" button to select an image to be used as an example for the blur and brightness check.

9. Start sorting
- Click the "Start" button to start the sorting process.
- To start the person search, you must first select a face using the "Search person" button.
- All images that meet the requirements will be sorted into the destination folder by year and month.
- The images are only copied, not moved or deleted.

10. Face recognition
- Click the "Search person" button to enable face recognition.
- Select a photo that contains a face. The program will automatically detect and mark the faces.
- Click on the detected face to select it. The selected face will be marked in red.
- Start the sorting process by clicking the "Start" button.

11. Search for photos with the selected face
- After starting the "Person Search", the program will search the source folder for photos containing the selected face.
- The found photos will be sorted into the destination folder by year and month.

Notes
- Make sure the EXIF data of the photos is correct, as the program uses this data to determine the creation date of the photos.
- The program creates a log of the actions performed in the file sort_photos_log.txt.
- Blur and brightness check doesn't work with face recognition.
"""

    help_text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10, bg="#F5F5DC", font=normal_font)
    help_text_widget.pack(fill=tk.BOTH, expand=True)

    help_text_widget.insert(tk.END, help_text)

    # Apply bold font to headings
    help_text_widget.tag_configure("bold", font=bold_font)
    help_text_widget.tag_add("bold", "2.0", "2.24") # 1. 
    help_text_widget.tag_add("bold", "6.0", "6.37") # 2. 
    help_text_widget.tag_add("bold", "10.0", "10.36") # 3. 
    help_text_widget.tag_add("bold", "14.0", "14.30") # 4. 
    help_text_widget.tag_add("bold", "19.0", "19.19") # 5. 
    help_text_widget.tag_add("bold", "23.0", "23.21") # 6. 
    help_text_widget.tag_add("bold", "27.0", "27.23") # 7. 
    help_text_widget.tag_add("bold", "30.0", "30.25") # 8. 
    help_text_widget.tag_add("bold", "33.0", "33.21") # 9. 
    help_text_widget.tag_add("bold", "39.0", "39.21") # 10. 
    help_text_widget.tag_add("bold", "45.0", "45.49") # 11. 
    help_text_widget.tag_add("bold", "49.0", "49.8") # Notes

    help_text_widget.config(state=tk.DISABLED)

def create_gui():
    global selected_image_path, example_image_label, middle_threshold_scale, source_folder_entry, destination_folder_entry, min_width_scale, min_height_scale, min_file_size_scale, check_blur_var, blur_threshold_scale, check_brightness_var, copy_non_conforming_var, include_subfolder_name_var, root, check_screenshot_var

    root = tk.Tk()
    root.title("Sort Photos by date")
    root.geometry("1280x740")

    selected_image_path = tk.StringVar()
    check_screenshot_var = tk.BooleanVar()
    
    # Folder selection
    def create_label_entry_button(row, label_text, entry_var, button_text, button_command): 
        tk.Label(root, text=label_text).grid(row=row, column=0, padx=20, pady=1, sticky="w")
        entry = tk.Entry(root, textvariable=entry_var, width=50)
        entry.grid(row=row, column=1, padx=1, pady=2, sticky="w")
        tk.Button(root, text=button_text, width=15, command=button_command).grid(row=row, column=2, padx=10, columnspan=2, pady=3, sticky="sw")
        return entry
  
    source_folder_entry = create_label_entry_button(0, "Source folder:", tk.StringVar(), "Browse", lambda: select_folder(source_folder_entry))
    destination_folder_entry = create_label_entry_button(1, "Destination folder:", tk.StringVar(), "Browse", lambda: select_folder(destination_folder_entry))
    
    # Include subfolder name
    include_subfolder_name_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Include subfolder names", variable=include_subfolder_name_var).grid(row=2, column=0, padx=20, columnspan=6, pady=1, sticky="sw")

    # Non conforming files
    copy_non_conforming_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Copy non-conforming images separately", variable=copy_non_conforming_var).grid(row=3, column=0, padx=20, columnspan=6, pady=1, sticky="w")

    # Checkbox screenshot detection
    check_screenshot_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Copy screenshots only", variable=check_screenshot_var).grid(row=4, column=0, padx=20, columnspan=6, pady=1, sticky="nw")
    
    # Face recon frame
    gesichtserkennung_frame = tk.LabelFrame(root, text="Example image / Target face", padx=4, pady=2)
    gesichtserkennung_frame.grid(row=2, column=2, padx=0, columnspan=6, rowspan=6, pady=10, sticky="nw")
    
    # Set the minimum size of the frame
    gesichtserkennung_frame.grid_propagate(False)  
    gesichtserkennung_frame.config(width=813, height=624) 
    
    # Example image
    tk.Button(root, text="Select example image", width=20, command=select_image).grid(row=0, column=3, padx=0, columnspan=2, pady=3, sticky="sw")
    example_image_label = tk.Label(gesichtserkennung_frame)
    example_image_label.grid(row=0, column=0, columnspan=4, rowspan=6, padx=0, pady=0, sticky="nw")

    # Start-Button
    tk.Button(root, text="Start", width=20, command=start_sorting).grid(row=1, column=3, padx=0, columnspan=2, pady=3, sticky="sw")
    
    # Person Search Button
    tk.Button(root, text="Search face", width=16, command=target_face).grid(row=0, column=4, padx=24, columnspan=2, pady=3, sticky="sw")

    # Help Button
    tk.Button(root, text="Help", width=16, command=show_help).grid(row=1, column=4, padx=24, columnspan=2, pady=3, sticky="sw")

    # Image specification frame
    bildanforderungen_frame = tk.LabelFrame(root, text="Image specification", padx=5, pady=5)
    bildanforderungen_frame.grid(row=6, column=0, columnspan=4, rowspan=4, padx=20, pady=10, sticky="nw")

    # Sliders
    def create_scale(frame, row, label, from_, to, command, length=250, orient=tk.HORIZONTAL, troughcolor=ENABLED_COLOR):
        scale = tk.Scale(frame, from_=from_, to=to, orient=orient, label=label, length=length, command=command, troughcolor=troughcolor)
        scale.grid(row=row, column=0, columnspan=3, padx=20, pady=0, sticky="nw")
        return scale

    min_width_scale = create_scale(bildanforderungen_frame, 0, "Min width", 0, MAX_WIDTH, lambda x: update_example_image())
    min_height_scale = create_scale(bildanforderungen_frame, 1, "Min height", 0, MAX_HIGH, lambda x: update_example_image())
    min_file_size_scale = create_scale(bildanforderungen_frame, 2, "Min file size (KB)", 0, MAX_SIZE, lambda x: update_example_image())
    
    # Empty row
    tk.Label(bildanforderungen_frame, text="").grid(row=3, column=0, pady=5)
   
    # Blur check
    check_blur_var = tk.BooleanVar()
    tk.Checkbutton(bildanforderungen_frame, text="Blur check (BETA)", variable=check_blur_var, command=on_check_blur).grid(row=4, column=0, padx=20, pady=5, sticky="sw")
    blur_threshold_scale = create_scale(bildanforderungen_frame, 5, "Blur threshold", 0, 100, lambda x: update_example_image())

    # Brightness check
    check_brightness_var = tk.BooleanVar()
    tk.Checkbutton(bildanforderungen_frame, text="Brightness check (BETA)", variable=check_brightness_var, command=lambda: update_example_image()).grid(row=6, column=0, padx=20, pady=5, sticky="sw")
    middle_threshold_scale = create_scale(bildanforderungen_frame, 7, "Middle brightness threshold", 0, 255, lambda x: update_example_image())
    middle_threshold_scale.set(100)
    
    # Sliders and checkboxes
    min_width_scale.config(command=lambda x: update_example_image())
    min_height_scale.config(command=lambda x: update_example_image())
    min_file_size_scale.config(command=lambda x: update_example_image())
    blur_threshold_scale.config(command=lambda x: update_example_image())
    middle_threshold_scale.config(command=lambda x: update_example_image())
    check_blur_var.trace_add('write', on_check_blur)
    check_brightness_var.trace_add('write', on_check_brightness)
    check_screenshot_var.trace_add('write', on_check_screenshot)

    # Initialize the sliders as disabled and grayed out
    configure_scale(blur_threshold_scale, tk.DISABLED, DISABLED_COLOR)
    configure_scale(middle_threshold_scale, tk.DISABLED, DISABLED_COLOR)

    root.mainloop()

# main
if __name__ == "__main__":
    example_image_label = None
    selected_image_path = None
    source_folder_entry = None
    destination_folder_entry = None
    min_width_scale = None
    min_height_scale = None
    min_file_size_scale = None
    check_blur_var = None
    blur_threshold_scale = None
    check_brightness_var = None
    copy_non_conforming_var = None
    progress_var = None
    progress_label = None
    root = None
    selected_face_location = None
    photo_path = None

    create_gui()