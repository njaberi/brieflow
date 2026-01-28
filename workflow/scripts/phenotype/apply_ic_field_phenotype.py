from tifffile import imread, imwrite

from lib.shared.illumination_correction import apply_ic_field

# load raw image data
raw_image_data = imread(snakemake.input[0])

# load illumination correction field
ic_field = imread(snakemake.input[1])

# load background image
background = imread(snakemake.input[2])

# apply illumination correction field
corrected_image_data = apply_ic_field(raw_image_data, correction=ic_field, background=background)

# save corrected image data
imwrite(snakemake.output[0], corrected_image_data)
