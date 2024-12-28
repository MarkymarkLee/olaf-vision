if [ ! -d models ]; then
    mkdir models
fi

rm -rf models/diffusion

# new
gdown --fuzzy https://drive.google.com/file/d/10UXBRf2-UL2o3hApQcgnUGNng4T7c50p/view?usp=sharing -O diffusion_model.zip

# old
# gdown --fuzzy https://drive.google.com/file/d/1ANpX5WgU4rMIkm-lx8RDANy-d4RXBOse/view?usp=sharing -O diffusion_model.zip
unzip diffusion_model.zip -d models
rm diffusion_model.zip

