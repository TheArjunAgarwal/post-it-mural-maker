import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def rgb_to_lab(rgb):
    """
    Converts RGB (0-255) to CIE Lab color space.
    """
    rgb = np.array(rgb, dtype=float) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100

    matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    xyz = np.dot(rgb, matrix.T)

    xyz_ref = np.array([95.047, 100.0, 108.883])
    xyz = xyz / xyz_ref
    
    mask = xyz > 0.008856
    xyz[mask] = np.power(xyz[mask], 1/3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

    L = (116 * xyz[..., 1]) - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])

    return np.stack([L, a, b], axis=-1)

def generate_post_it_plan(image_path, mural_width, mural_height, available_colors):
    # 1. Load and resize
    img = Image.open(image_path).convert('RGB')
    img_small = img.resize((mural_width, mural_height), resample=Image.LANCZOS)
    
    # --- PRE-DITHERING SHARPNESS ---
    enhancer = ImageEnhance.Sharpness(img_small)
    img_small = enhancer.enhance(2.5)  # Increase sharpness factor
    
    pixels_rgb = np.array(img_small)

    # 2. Convert to Lab Space
    print("Converting to CIE Lab space...")
    pixels_lab = rgb_to_lab(pixels_rgb)
    
    color_names = list(available_colors.keys())
    color_rgb_values = np.array(list(available_colors.values()))
    color_lab_values = rgb_to_lab(color_rgb_values)

    # 3. Bayer Dithering (4x4 Matrix)
    # The Bayer matrix M defines the threshold levels
    bayer_matrix = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ]) / 16.0
    
    # Scale of the dither effect (adjust this for more/less texture)
    spread = 15.0 

    height, width, _ = pixels_lab.shape
    mural_indices = np.zeros((height, width), dtype=int)

    print("Applying Bayer ordered dithering...")
    for y in range(height):
        for x in range(width):
            # Calculate threshold offset based on position
            threshold = bayer_matrix[y % 4, x % 4] - 0.5
            
            # Apply offset to the L (Lightness) channel
            pixel_val = pixels_lab[y, x].copy()
            pixel_val[0] += threshold * spread
            
            # Delta E distance to nearest color
            distances = np.sqrt(np.sum((color_lab_values - pixel_val)**2, axis=1))
            mural_indices[y, x] = np.argmin(distances)

    # 4. Create Visual Map (Legacy logic preserved)
    cell_size = 60
    output_img = Image.new('RGB', (width * cell_size, height * cell_size), color='white')
    draw = ImageDraw.Draw(output_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        font = ImageFont.load_default()

    for y in range(height):
        for x in range(width):
            color_idx = mural_indices[y, x]
            actual_rgb = tuple(color_rgb_values[color_idx])
            rect = [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size]
            draw.rectangle(rect, fill=actual_rgb, outline=(50, 50, 50))
            
            # num_text = str(color_idx + 1)
            # tw, th = draw.textsize(num_text, font=font)
            # draw.text((x * cell_size + (cell_size - tw) / 2, 
            #            y * cell_size + (cell_size - th) / 2), 
            #           num_text, fill=(0, 0, 0), font=font)

    legend = {i + 1: color_names[i] for i in range(len(color_names))}
    return output_img, legend

# --- CONFIGURATION (Unchanged) ---
my_colors = {
    "Canary Yellow": (255, 255, 150),
    "Neon Pink": (255, 100, 180),
    "Electric Blue": (100, 200, 255),
    "Lime Green": (150, 255, 100),
    "Orange": (255, 165, 0),
    "White": (255, 255, 255),
    "Black": (0, 0, 0)
}

final_map, color_sheet = generate_post_it_plan("/home/students/arjuna.ug2024/graph/starry-night.png", 64, 45, my_colors)
final_map.save("starry-night_mural_blueprint_bayer_small_wotext.png")

print("\nSuccess! Mural blueprint with Bayer dithering saved.")