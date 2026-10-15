import numpy as np
from PIL import Image, ImageDraw, ImageFont

def rgb_to_lab(rgb):
    """
    Converts RGB (0-255) to CIE Lab color space.
    """
    # 1. Normalize and Undo Gamma Correction (sRGB to Linear RGB)
    rgb = np.array(rgb) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100

    # 2. Linear RGB to XYZ (D65 Illuminant)
    matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    xyz = np.dot(rgb, matrix.T)

    # 3. XYZ to CIE Lab
    # Reference white for D65
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
    pixels_rgb = np.array(img_small)

    # 2. Convert everything to Lab Space for perceptual matching
    print("Converting image and post-it colors to CIE Lab space...")
    pixels_lab = rgb_to_lab(pixels_rgb)
    
    color_names = list(available_colors.keys())
    color_rgb_values = np.array(list(available_colors.values()))
    color_lab_values = rgb_to_lab(color_rgb_values)

    # 3. Map pixels to nearest Lab color
    height, width, _ = pixels_lab.shape
    mural_indices = np.zeros((height, width), dtype=int)

    for y in range(height):
        for x in range(width):
            pixel_val = pixels_lab[y, x]
            # Delta E calculation (Euclidean distance in Lab space)
            distances = np.sqrt(np.sum((color_lab_values - pixel_val)**2, axis=1))
            mural_indices[y, x] = np.argmin(distances)

    # 4. Create the Visual Map
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
            
            # Draw square
            rect = [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size]
            draw.rectangle(rect, fill=actual_rgb, outline=(50, 50, 50))
            
            # Draw Number (using legacy .textsize for your environment)
            num_text = str(color_idx + 1)
            tw, th = draw.textsize(num_text, font=font)
            draw.text((x * cell_size + (cell_size - tw) / 2, 
                       y * cell_size + (cell_size - th) / 2), 
                      num_text, fill=(0, 0, 0), font=font)

    legend = {i + 1: color_names[i] for i in range(len(color_names))}
    return output_img, legend

# --- CONFIGURATION ---
my_colors = {
    "Canary Yellow": (255, 255, 150),
    "Neon Pink": (255, 100, 180),
    "Electric Blue": (100, 200, 255),
    "Lime Green": (150, 255, 100),
    "Orange": (255, 165, 0),
    "White": (255, 255, 255),
    "Black": (0, 0, 0)
}

path = "red-wave"

# Run
final_map, color_sheet = generate_post_it_plan(f"/home/students/arjuna.ug2024/graph/{path}.png", 100, 70, my_colors)

# Save
final_map.save(f"{path}-mural_blueprint3.png")

print("\n--- POST-IT COLOR LEGEND ---")
for num, name in color_sheet.items():
    print(f"Number {num}: {name}")
print("\nSuccess! Mural blueprint saved as 'mural_blueprint.png'.")