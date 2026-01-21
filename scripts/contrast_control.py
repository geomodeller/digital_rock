
from skimage import exposure
import numpy as np
def contrast_stretch(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, (p_low, p_high))
    img_cs = np.clip(img, lo, hi)
    img_cs = (img_cs - lo) / (hi - lo + 1e-12)
    return (img_cs * 255).astype(np.uint8)


def contrast_clahe(img, clip_limit=0.02, nbins=256):
    """ CLAHE (local contrast enhancement)

    CLAHE = Contrast Limited Adaptive Histogram Equalization

    Why it works
    - Enhances contrast locally
    - Fixes spatially varying illumination
    - Very effective for heterogeneous rocks

    Caveat
    - Can amplify noise if used too early
    - Must be applied after denoising
    """

    img01 = img.astype(np.float32) / 255.0
    out = exposure.equalize_adapthist(img01, clip_limit=clip_limit, nbins=nbins)
    return (out * 255).astype(np.uint8)