import numpy as np
import mahotas as mh
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion


#~~~~~~~~~~~~~~~~~~~~~~~ functions for partition measurements ~~~~~~~~~~~~~~~~~~~~~
def get_pc(dense_antibody, dilute_antibody):
  #use dense and dilute signals to calculate PC if both finite and dilute not 0
  if np.isfinite(dense_antibody) and np.isfinite(dilute_antibody) and dilute_antibody != 0:
    pc = dense_antibody / dilute_antibody
  else:
    pc = np.nan
  return pc

def get_dense_dilute(img, mask_dense, mask_dilute):
  #dense and dilute pixels
  dense_pixels = img[mask_dense]
  dilute_pixels = img[mask_dilute]

  #get dense and dilute values, if the phases exist
  dense_antibody  = np.percentile(dense_pixels,90) if dense_pixels.size else np.nan
  dilute_antibody = np.percentile(dilute_pixels,50) if dilute_pixels.size else np.nan

  return dense_antibody, dilute_antibody

def norm_and_project(regionprop_object, channels, norm_perc=99.9,ignore_val=65535):
    """Normalize and project specified channels within a regionprops object.

    Args:
        regionprop_object: regionprops object with multichannel intensity_image
        channels: list of channel indices to project
        norm_perc: percentile for normalization
        ignore_val: val that is impossible/means 
                    something else (like pixels outside of aligned area),
                    will be excluded for max-norm before projection.

    Returns:
        2D projected image
    """

    to_project = []
    
    for channel in channels:
        img = regionprop_object.intensity_image[:,:,channel]

        valid_pixels = img[img < ignore_val]
        if valid_pixels.size > 0:
            max_sig = np.percentile(valid_pixels, norm_perc)
        else:
            max_sig = 0

        if max_sig > 0:
            normalized = np.clip(img / max_sig, 0, 1)
        else: #avoid division by 0
            normalized = np.zeros_like(img, dtype=float)
        to_project.append(normalized/len(channels))

    return np.sum(to_project, axis=0)

def get_masks_img(projected,region):
    """Get dense and dilute masks for a regionprops region within a projected image."""
    
    pixels = projected[region.image]
    if pixels.size==0:
        empty = np.zeros(region.image.shape, dtype=bool)
        return empty, empty

    #first pass otsu mask
    mask_dense = (projected > threshold_otsu(pixels)) & region.image
    mask_dense = mh.close_holes(mask_dense) & region.image

    #erode mask to get dense
    mask_erode = binary_erosion(mask_dense)
    
    #dilate mask to get dilute
    mask_dilate = binary_dilation(mask_dense)
    mask_dilate = ~mask_dilate & region.image

    return mask_erode, mask_dilate

def get_partition_metrics(regionprops_obj, testing=False):
    """Takes in a multichannel regionprops object (nucleus), calculates dense and dilute masks on projection of all channels.
    Then calculates partition metrics for each channel using these masks.
    Returns a dictionary. each metric is a key, and values are a list of that metric for each channel.
    

    Standalone function for notebook use. Returns dict of lists.
    
    Args:
        region: multichannel regionprops object
    """
    #get channels to project
    nchannels = regionprops_obj.intensity_image.shape[-1]
    all_channels = list(range(nchannels))

    #normalize and project across channels
    projected = norm_and_project(regionprops_obj, all_channels)
    
    #get masks from projection
    mask_dense, mask_dilute = get_masks_img(projected,regionprops_obj)

    #initialize storage lists
    dense_vals, dilute_vals, pc_vals, cv_vals = [],[],[],[] #lists to contain all channel info for this cell

    #loop over channels
    for channel in all_channels:
        skip = False
        cell_box = regionprops_obj.intensity_image[:,:,channel]

        #borders that are clipped in this channel have been set to 65535. 
        #which should not appear in normal images which are background subtracted.
        #can we get img coords of bbox?
        if np.any(cell_box == 65535):
            skip = True

        if skip:
            dense_vals.append(np.nan)
            dilute_vals.append(np.nan)
            pc_vals.append(np.nan)
            cv_vals.append(np.nan)
        
        #extract phenotypes
        else:
            #partitioning
            dense_antibody, dilute_antibody = get_dense_dilute(cell_box, mask_dense, mask_dilute)
            pc = get_pc(dense_antibody, dilute_antibody)

            #cv
            pixels = cell_box[regionprops_obj.image]
            cv = np.std(pixels)/np.mean(pixels) if pixels.size else np.nan
            
            #TODO other phenotypes --> integrated intensity (unique for each channel), 
            # area (Same for every channel?)

            #store data for each channel in regionprops_obj
            cv_vals.append(cv)
            dense_vals.append(dense_antibody)
            dilute_vals.append(dilute_antibody)
            pc_vals.append(pc)
    
    out_dict = {"dense_intensities": dense_vals,
            "dilute_intensities": dilute_vals,
            "coefficients of variation": cv_vals,
            "Partition Coefficients": pc_vals}

    if testing:
        out_dict['dense mask']=mask_dense
        out_dict['dilute mask'] = mask_dilute    

    return out_dict

def partition_features_wrapper(r):
    """Wrapper for feature_table_multichannel compatibility.
    
    Returns flattened array: [pc_ch0, pc_ch1, ..., dense_ch0, ..., dilute_ch0, ..., cv_ch0, ...]
    """
    metrics = get_partition_metrics(r)
    return np.array([
        metrics["dense_intensities"],
        metrics["dilute_intensities"],
        metrics["Partition Coefficients"],
        metrics["coefficients of variation"],
    ]).flatten()

#~~~~~~~~~~~~~~~~~~~~~~~ feature and column mappers for partition measurements ~~~~~~~~~~~~~~~~~~~~~
partition_features_multichannel = {
    "partition": partition_features_wrapper, 
}

partition_columns_multichannel = {
    "partition": [
        "dense_intensity",
        "dilute_intensity",
        "partition_coefficient",
        "coefficient_of_variation",
    ],
}