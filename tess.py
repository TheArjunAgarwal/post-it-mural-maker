import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_post_it_plan(image_path, mural_width, mural_height, available_colors):
    """
    image_path: Path to your input image
    mural_width: Number of post-its wide
    mural_height: Number of post-its high
    available_colors: Dict of { "Color Name": (R, G, B) }
    """
    
    # 1. Load and resize image to the grid dimensions
    img = Image.open(image_path).convert('RGB')
    # This works in both old and new versions of Pillow
    img_small = img.resize((mural_width, mural_height), resample=Image.LANCZOS)
    pixels = np.array(img_small)

    # Convert available colors to a searchable format
    color_names = list(available_colors.keys())
    color_values = np.array(list(available_colors.values()))

    # 2. Map each pixel to the nearest available post-it color
    # We use Euclidean distance in RGB space
    height, width, _ = pixels.shape
    mural_indices = np.zeros((height, width), dtype=int)

    for y in range(height):
        for x in range(width):
            pixel = pixels[y, x]
            # Calculate distance to all available colors
            distances = np.sqrt(np.sum((color_values - pixel)**2, axis=1))
            mural_indices[y, x] = np.argmin(distances)

    # 3. Create the Visual Map (High-res grid with numbers)
    cell_size = 60  # Size of each "post-it" in the output image
    output_img = Image.new('RGB', (width * cell_size, height * cell_size), color='white')
    draw = ImageDraw.Draw(output_img)
    
    try:
        # Attempt to load a clear font, fallback to default if not found
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        font = ImageFont.load_default()

    for y in range(height):
        for x in range(width):
            color_idx = mural_indices[y, x]
            color_rgb = tuple(color_values[color_idx])
            
            # Draw the colored square
            rect = [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size]
            draw.rectangle(rect, fill=color_rgb, outline=(50, 50, 50))
            
            # Draw the number (color index + 1)
            num_text = str(color_idx + 1)
            # Center the text in the square
            # Legacy way (Works on older Pillow versions)
            tw, th = draw.textsize(num_text, font=font)
            draw.text((x * cell_size + (cell_size - tw) / 2, 
                       y * cell_size + (cell_size - th) / 2), 
                      num_text, fill=(0, 0, 0), font=font)

    # 4. Generate the Legend
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
    "Black": (0,0,0)
}

# Run the generator
# Dimensions: 30 notes wide, 20 notes high
final_map, color_sheet = generate_post_it_plan("/home/students/arjuna.ug2024/graph/image2.png", 128, 72, my_colors)

# Save and Output
final_map.save("mural_blueprint3-1.png")
print("--- POST-IT COLOR LEGEND ---")
for num, name in color_sheet.items():
    print(f"Number {num}: {name}")