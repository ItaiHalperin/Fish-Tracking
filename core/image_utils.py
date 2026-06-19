import numpy as np

def crop_with_padding(
    image: np.ndarray, 
    box: list[float] | np.ndarray, 
    padding: float = 0.10, 
    min_size: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Crops a bounding box from an image with a fractional padding.
    
    Parameters
    ----------
    image: np.ndarray
        The source image (H, W, C).
    box: list[float] | np.ndarray
        The bounding box in [x1, y1, x2, y2] format.
    padding: float
        The fractional padding to add around the box (e.g., 0.10 for 10%).
    min_size: tuple[int, int] | None
        If provided and the resulting crop is empty (size 0), returns a black image 
        of this size (W, H). If None, an empty crop can be returned.
        
    Returns
    -------
    np.ndarray
        The cropped image patch.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    
    w = x2 - x1
    h = y2 - y1
    
    pad_w = w * padding
    pad_h = h * padding
    
    cx1 = max(0, int(x1 - pad_w))
    cy1 = max(0, int(y1 - pad_h))
    cx2 = min(w_img, int(x2 + pad_w))
    cy2 = min(h_img, int(y2 + pad_h))
    
    crop = image[cy1:cy2, cx1:cx2]
    
    if crop.size == 0 and min_size is not None:
        return np.zeros((min_size[1], min_size[0], 3), dtype=np.uint8)
        
    return crop
