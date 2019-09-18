from __future__ import division

import os
import tempfile

from fpdf import FPDF
from PIL import Image


ORANGE = (255, 128, 0)
PURPLE = (153, 51, 255)
YELLOW = (255, 215, 0)
GREEN = (0, 215, 0)
WHITE = (255, 255, 255)
CYAN = (0, 215, 215)

def cm2in(length_cm):
    """Convert the given centimeter length to inches.

    :param length_cm: The length to convert in centimeters
    :return: The given length in inches [float]
    """
    return length_cm / 2.54


def in2cm(length_in):
    """Convert the given inch length to centimeters.

    :param length_in: The length to convert in inches
    :return: The given length in centimeters [float]
    """
    return 2.54 * length_in


def px2in(length_p, dpi=80):
    """Convert the given pixel length to inches.

    :param length_p: The length to convert in pixels
    :param dpi: The number of pixels per inch
    :return: The given length in inches [float]
    """
    return length_p / dpi


def in2px(length_i, dpi=80):
    """Convert the given inch length to pixels.

    :param length_i: The length to convert in inches
    :param dpi: The number of pixels per inch
    :return: The given length in pixels [float]
    """
    return dpi * length_i


def px2cm(length_p, dpi=80):
    """Convert the given pixel length to centimeters.

    :param length_p: The length to convert in pixels
    :param dpi: The number of pixels per inch
    :return: The given length in centimeters [float]
    """
    return in2cm(px2in(length_p, dpi=dpi))


def cm2px(length_cm, dpi=80):
    """Convert the given centimeter length to pixels.

    :param length_cm: The length to convert in centimeters
    :param dpi: The number of pixels per inch
    :return: The given length in pixels [float]
    """
    return dpi * cm2in(length_cm)


def add_image_to_pdf(pdf, image_path, x_cm, y_cm, w_cm, h_cm, b_cm=0, color=None):
    """

    :param pdf: The FPDF instance to add to
    :param image_path: Path to the image file
    :param x_cm: x-coordinate of the top-left corner in cm
    :param y_cm: y-coordinate of the top-left corner in cm
    :param w_cm: Width of the image on the PDF in cm
    :param h_cm: Height of the image on the PDF in cm
    :param b_cm: Thickness of the border in cm
    :param color: Tuple indicating the color [(R, G, B) with values in 0-255]
    """

    if color is not None and not isinstance(color, tuple):
        raise ValueError('color must either be None or a tuple')

    # Draw the border
    if b_cm > 0 and color is not None:
        pdf.set_draw_color(*color)
        pdf.set_line_width(2 * b_cm)
        pdf.rect(x_cm, y_cm, w_cm, h_cm)
    # Draw the image
    pdf.image(image_path, x_cm, y_cm, w_cm, h_cm)


def add_cropped_image_to_pdf(pdf, image_path, region, x_cm, y_cm, w_cm, h_cm, b_cm=0, color=None):
    """

    :param pdf: The FPDF instance to add to
    :param image_path: Path to the image file
    :param region: The part of the image to draw, specified as four decimal values specifying the top-left and
                   bottom-right coordinates of the full image
    :param x_cm: x-coordinate of the top-left corner in cm
    :param y_cm: y-coordinate of the top-left corner in cm
    :param w_cm: Width of the image on the PDF in cm
    :param h_cm: Height of the image on the PDF in cm
    :param b_cm: Thickness of the border in cm
    :param color: Tuple indicating the color [(R, G, B) with values in 0-255]
    """

    if color is not None and not isinstance(color, tuple):
        raise ValueError('color must either be None or a tuple')

    # Get extension of given image path
    _, ext = os.path.splitext(image_path)
    # Create the cropped image
    with tempfile.NamedTemporaryFile(suffix=ext) as temp_file:
        full_image = Image.open(image_path)
        h, w = full_image.size
        crop_coords = [h * region[0], w * region[1], h * region[2], w * region[3]]
        crop_coords = [int(x) for x in crop_coords]
        cropped_image = full_image.crop(crop_coords)
        cropped_image.save(temp_file.name)

        add_image_to_pdf(pdf, temp_file.name, x_cm, y_cm, w_cm, h_cm, b_cm=b_cm, color=color)


def add_text_to_pdf(pdf, text, x_cm, y_cm, font_size_pt):
    """

    :param pdf:
    :param text:
    :param x_cm:
    :param y_cm:
    :param font_size_pt:
    """

    # Store previous font size
    old_font_size = pdf.font_size_pt
    # Draw the text
    pdf.set_font_size(font_size_pt)
    pdf.text(x_cm, y_cm + .7 * in2cm(font_size_pt / 72), text)
    # Restore previous font size
    pdf.set_font_size(old_font_size)


def get_text_width(text, font_size_pt):
    """Obtain the length of the given string as it appears in the given document

    :param text: str
    :param font_size_pt: Font size in pt units
    :return: The PDF width of the string
    """

    # Create function instance of FPDF if non-existent
    if 'pdf' not in get_text_width.__dict__:
        get_text_width.pdf = create_pdf(0, 0, 'cm')

    # Store previous font size
    old_font_size = get_text_width.pdf.font_size_pt
    # Get text width given new font size
    get_text_width.pdf.set_font_size(font_size_pt)
    ret = get_text_width.pdf.get_string_width(text)
    # Restore previous font size
    get_text_width.pdf.set_font_size(old_font_size)
    return ret


def create_pdf(width, height, units='cm'):
    if units not in ['cm', 'in']:
        raise ValueError('units argument must be "cm" or "in"')

    if units == 'in':
        width_cm = in2cm(width)
        height_cm = in2cm(height)
    else:
        width_cm = width
        height_cm = height

    # Set size and units of PDF, and generate the first page
    pdf = FPDF(unit='cm', format=(width_cm, height_cm))
    pdf.add_page()
    # Set font and font size
    pdf.set_font('Times')

    return pdf

