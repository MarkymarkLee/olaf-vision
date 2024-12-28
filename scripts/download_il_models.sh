# This file downloads the trained bc-models from Google Drive and unzips them.
# These models are trained with the meta-world dataset.
# The models will be put in a folder called "outputs" in the root directory of the project.
# The models are not used in the final experiments and are not necessary to run the code.

gdown --fuzzy https://drive.google.com/file/d/1XngzN43pGxFgXN04ceDcawHnceDTtJ54/view?usp=drive_link -O bc-models.zip
unzip bc-models.zip
rm -f bc-models.zip