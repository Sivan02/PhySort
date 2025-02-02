This is my first GitHub repository. I was looking for a tool to automatically sort photo albums. Since I couldn't find anything suitable
and I'm currently learning Python, I decided to build one myself. 
 
The program sorts photos by date (year, month). You can configure various settings to determine which photos should be included and which should not.
It is also possible to sort photos by facess. Files are only copied, not deleted. For more info, start the program and klick on "Help".

Features
Sort by Date: Photos are sorted into folders by year and month based on their EXIF data.
Blur Check (BETA): Checks images for blur and only copies sharp images.
Brightness Check (BETA): Checks images for poor lighting and only copies well-lit images.
Screenshot Detection: Copies only screenshots based on filenames and EXIF data.
Face Recognition: Detects and marks faces in images. Allows searching and copying photos containing a specific face.
Non-Conforming Images: Option to copy images that do not meet the specified requirements into a separate folder.

Ensure Python 3.6 or higher is installed.
Install the required libraries using the provided batch script:
install_requirements.bat

