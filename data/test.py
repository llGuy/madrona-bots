from PIL import Image

def change_green_to_orange(input_image_path, output_image_path):
    # Open an image file
    with Image.open(input_image_path) as img:
        # Convert image to RGBA (if not already in RGBA)
        img = img.convert("RGBA")
        # Get the data of the image
        data = img.getdata()

        new_data = []
        # Define the color range for green and the orange replacement color
        green_range = (100, 100, 200)
        orange_color = (255, 255, 255, 255)

        for item in data:
            # print(item)
            # Change all green (or shades of green) pixels to orange
            if item[0] < green_range[0] and item[1] < green_range[1] and item[2] > green_range[2]:
                new_data.append(orange_color)
            else:
                new_data.append(item)

        # Update image data
        img.putdata(new_data)
        # Save the image
        img.save(output_image_path)

# Usage example
input_image_path = 'new_smile.png'
output_image_path = 'new_smile2.png'
change_green_to_orange(input_image_path, output_image_path)
