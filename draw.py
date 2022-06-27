from PIL import Image, ImageDraw
import pdb
def draw_ellipse(image, bounds, width=1, outline='white', antialias=4):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""

    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results. 
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')

    draw = ImageDraw.Draw(mask)
    #pdb.set_trace();

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.rectangle([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS 
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)

# green background image
image = Image.new(mode='RGB', size=(700, 300), color='green')

ellipse_box = [50, 50, 300, 250]

# draw a thick white ellipse and a thin black ellipse
draw_ellipse(image, ellipse_box, width=10,outline='red')

# draw a thin black line, using higher antialias to preserve finer detail
#draw_ellipse(image, ellipse_box, outline='black', width=.5, antialias=8)

# Lets try without antialiasing


image.show()