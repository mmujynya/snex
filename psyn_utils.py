import json
import os
import supernotelib as sn
import base64
from unidecode import unidecode
from PIL import Image,  ImageDraw
import numpy as np
import copy
from datetime import datetime
import uuid
from intervaltree import IntervalTree
import re
from shapely.geometry import LineString, MultiLineString
import random

UNSTITCH = False
DEFAULT_PEN = 10  # 10 is Needle-Point
DEFAULT_PEN_SIZE = "0.3"
DEFAULT_PRESSURE = 2000
HIGHLIGHTS_COLOR = {
    'light gray': (201, 201, 201),
    'dark gray': (157, 157, 157),
    'white': (254, 254, 254),
    'black': (0, 0, 0)}
OFFSET_T_X = 16
OFFSET_T_Y = 32
PYSN_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__))
CONFIG = os.path.join(PYSN_DIRECTORY, 'config')
MAX_HORIZONTAL_PIXELS_N6 = 1404        # Default pixels size on A6x2. You may have to change that for other devices
MAX_VERTICAL_PIXELS_N6 = 1872          # Default pixels size on A6x2. You may have to change that for other devices
MAX_HORIZONTAL_PIXELS_N5 = 1920
MAX_VERTICAL_PIXELS_N5 = 2560
ADB_SCREEN_MAX_X_N6 = 11864    # Inferred from binaries;I previously estimated this to be 11800
ADB_SCREEN_MAX_X_N5 = 16224   # For N5 series like Manta
ADB_SCREEN_MAX_Y_N6 = 15819    # I previously estimated this to be 15800
ADB_SCREEN_MAX_Y_N5 = 21632  # For N5 series like Manta
ADB_SCREEN_OFFSET = 5
CONTOUR_RESOLUTION = 2
NOTEBOOK_DEVICE = 'N5'
FONTS_NARROWING_AFTER = {'F': 0.5, 'f': 0.5, 'i': 0.5, 'I': 0.5, 'l': 0.5}
FONTS_NARROWING_BEFORE = {'j': 0.5}
FONT_NAME = "barlow"
FONT_RATIOS = [0.8, 0.55]
TEXT2NOTES_STYLING = {
        "$page": {
            "next_word_count": 1,
            "ord": 170,
            "weight": 0.4,
            "toc_bg_color": 202
        },
        "^h1": {
            "ord": 170,
            "weight": 0.6,
            "toc_bg_color": 254,
            "indent": 4,
            "creturn": 4,
            "size": 0.8,
            "case": "upper"
        },
        "^h2": {
            "ord": 171,
            "weight": 0.4,
            "toc_bg_color": 254,
            "indent": 3,
            "creturn": 3
        },
        "^h3": {
            "ord": 169,
            "weight": 0.2,
            "toc_bg_color": 254,
            "indent": 2,
            "creturn": 2
        },
        "^h4": {
            "ord": 169,
            "weight": 0.2,
            "toc_bg_color": 254,
            "indent": 1,
            "creturn": 1
        },
        "^l1": {
            "prefix": "-",
            "indent": -2,
            "creturn": 1
        },
        "^l2": {
            "prefix": "+",
            "indent": -4,
            "creturn": 1
        },
        "^b1": {
            "weight": 0.4
        }
    }
CHAR_V_OFFSET = {
            "g": 15,
            "p": 15,
            "q": 15,
            "y": 15,
            "f": 15,
            "'": -25,
            "-": -15,
            ",": 15,
            ";": 15,
            "“": -30,
            "”": -30,
            "’": -30,
            "–": -30,
            "\"": -30,
            "^": -30,
            "ª": 40,
            "°": 10}                              # (!) Vertical offset for some fonts
MAX_PEN_TO_STROKE_LOOP = 500
NEEDLE_POINT_SIZES = {
    "0.4": 500,
    "0.1": 200,
    "0.2": 300,
    "0.3": 400,
    "0.5": 700,
    "0.6": 800,
    "0.7": 1000,
    "0.8": 1100,
    "0.9": 1200,
    "1": 1300,
    "2": 2400
}
MARGIN_LEFT_SIZE = 0
BLANK_TEMPLATE_PICTURE2 = os.path.join(
    CONFIG, 'blankmanta.png')
SPECIAL_LENGTH = 0x4000             # (*)
SPECIAL_LENGTH_FOR_BLANK = 0x400    # (*)
SPECIAL_LENGTH_MARKER = 0xff        # (*)
COLORCODE_BACKGROUND = 0x62         # (*)
COLORCODE_WHITE = 0x65              # (*)
ASCII_X0 = 111
ASCII_X02 = 152
ASCII_SIDE = 118
ASCII_SIDE2 = 161
CUSTOM_MAP = {
    "Ç": 128,
    "ü": 129,
    "é": 130,
    "â": 131,
    "ä": 132,
    "à": 133,
    "å": 134,
    "ç": 135,
    "ê": 136,
    "ë": 137,
    "è": 138,
    "ï": 139,
    "î": 140,
    "ì": 141,
    "Ä": 142,
    "Å": 143,
    "É": 144,
    "æ": 145,
    "Æ": 146,
    "ô": 147,
    "ö": 148,
    "ò": 149,
    "û": 150,
    "ù": 151,
    "ÿ": 152,
    "Ö": 153,
    "Ü": 154,
    "¢": 155,
    "£": 156,
    "¥": 157,
    "₧": 158,
    "ƒ": 159,
    "á": 160,
    "í": 161,
    "ó": 162,
    "ú": 163,
    "ñ": 164,
    "Ñ": 165,
    "ª": 166,
    "º": 167,
    "¿": 168,
    "•": 172,
    "○": 173,
    "■": 174,
    "□": 175}
DEBUG_MODE = False
FILE_RECOGN_LANGUAGE = 'en_US'
BLANK_TEMPLATE_PICTURE = os.path.join(
    CONFIG, 'blanknomad.png')
SCALE_RATIO_TEXT2NOTE = 1
DEFAULT_PEN_COLOR = "black"
FONTS_FAMILY_MAP = {
    6: "barlow",
    5: "segoe",
    8: "calibri"
}


def list_to_tree(interval_list):
    """ Load a tree from JSON"""
    tree = IntervalTree()
    for interval in interval_list:
        tree[interval["begin"]:interval["end"]] = interval["data"]
    return tree


def font_size_to_pixels(font_size, dpi=300):
    """ Converts a font size in points into pixels
        1pt = 1/72 of an inch """
    return font_size*dpi/72


def get_key_by_value(color_dict, value):
    for key, val in color_dict.items():
        print(f'----{key}={val}')
        if val == value:
            return key
    return None  # Return None if value not found


def contains_hex_sequence(binary_data: bytes, hex_seq: str) -> bool:
    """
    Check if the given hex sequence exists in binary_data.

    Parameters:
        binary_data (bytes): The binary data to search.
        hex_seq (str): The hex sequence (e.g., "deadbeef").

    Returns:
        bool: True if the sequence exists, False otherwise.
    """
    pattern = bytes.fromhex(hex_seq)
    return pattern in binary_data


def is_n6_file(binary_data):
    return contains_hex_sequence(binary_data, '3C4150504C595F45515549504D454E543A4E36')


def decimal_ieee754_from_binary(binary_data, position, offset=0, num_bytes=4):
    """ Extract 4 bytes starting from the specified position """
    try:
        if position + num_bytes > len(binary_data):
            return None
        bytes_data = binary_data[position:position + num_bytes]
        bytes_data = bytes_data.hex()

        byte1 = bytes_data[0:2]
        byte2 = bytes_data[2:4]
        byte3 = bytes_data[4:6]
        byte4 = bytes_data[6:8]

        # Convert each byte to binary string, keeping it 8 bits long
        byte1str = format(int(byte1, 16), '08b')
        byte2str = format(int(byte2, 16), '08b')
        byte3str = format(int(byte3, 16), '08b')
        byte4str = format(int(byte4, 16), '08b')

        # Concatenate binary representations in the custom order
        binary_str = byte4str + byte3str + byte1str + byte2str

        # Extract sign (1 bit), exponent (8 bits), and mantissa (23 bits)
        sign = int(binary_str[0], 2)
        exponent = int(binary_str[1:9], 2)
        mantissa_str = binary_str[9:]

        # Convert mantissa to decimal using the formula in Excel
        mantissa = 1 + sum(int(mantissa_str[i]) * 2**-(i + 1) for i in range(23))

        # Compute the final floating point value
        result = ((-1) ** sign) * (2 ** (exponent - 127)) * mantissa

        # Apply the scaling factor
        final_value = result + offset

        return final_value
    except Exception as e:
        print(f'*** decimal_ieee754_from_binary: {e} - len bin: {len(binary_data)} - pos: {position} - num_bytes: {num_bytes} - offset: {offset}')
        return None


def read_endian_int_at_position(data, position, num_bytes=4, endian='little'):
    """ Returns the endian integer equivalent of 'num_bytes' read
        from 'data' at 'position' """
    completed = False
    integer_value = 0

    try:
        # Ensures valid endian type
        if endian not in ['little', 'big']:
            raise ValueError("Endian must be 'little' or 'big'")

        # Ensure the position and number of bytes to read are within the data range
        if position < 0 or position + num_bytes > len(data):
            raise ValueError("Invalid position or number of bytes to read")

        # Read the specified number of bytes from the given position
        byte_sequence = data[position:position + num_bytes]
        # print(byte_sequence.hex().upper())

        # Convert the byte sequence from little-endian to an integer
        integer_value = int.from_bytes(byte_sequence, byteorder=endian)
        completed = True
    except Exception as e:
        print(f'*** Error @ read_endian_int_at_position: {e}')
        print(f'>>> position: {position} - num_bytes: {num_bytes} - endian: {endian}')

    return completed, integer_value


def int_to_little_endian_bytes(value, num_bytes=4, byteorder='little'):
    # Ensures valid endian type
    if byteorder not in ['little', 'big']:
        raise ValueError("Endian must be 'little' or 'big'")
    # Ensure the value fits within the specified number of bytes
    if value < 0 or value >= (1 << (num_bytes * 8)):
        raise ValueError(f"Value {value} out of range for {num_bytes} bytes")

    # Convert the integer to a byte sequence in the specified endian format
    return value.to_bytes(num_bytes, byteorder=byteorder)


def series_bounds(series):
    """ Returns screen bounds based on device series """
    # print(f'       series:{series}')
    if series in ['N5']:
        return [
            MAX_HORIZONTAL_PIXELS_N5, MAX_VERTICAL_PIXELS_N5,
            ADB_SCREEN_MAX_X_N5, ADB_SCREEN_MAX_Y_N5]
    else:
        return [
            MAX_HORIZONTAL_PIXELS_N6, MAX_VERTICAL_PIXELS_N6,
            ADB_SCREEN_MAX_X_N6, ADB_SCREEN_MAX_Y_N6]


def read_json(afilename):
    """ Reading a json from a filename """
    try:
        with open(afilename, 'r', encoding='utf-8') as file:
            a_json_text = file.read().strip().replace("\ufeff", "")
            return json.loads(a_json_text)
    except Exception as e:
        print()
        print(f'*** read_json: {e}')
        return {}


def save_json(afilename, ajson):
    """ Saving a json with identation """
    with open(afilename, 'w') as file:
        file.write(json.dumps(ajson, indent=4))


def topright_to_topleft(a_reversed_point, max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y):
    """ Screen coordinates conversion
        Warning: reverts back to x & y """
    try:
        a_point = (
            max(min(
                round((adb_screen_max_x+OFFSET_T_X-a_reversed_point[1])*max_horizontal_pixels/adb_screen_max_x), max_horizontal_pixels), 0),
            max(min(round((a_reversed_point[0]+OFFSET_T_Y)*max_vertical_pixels/adb_screen_max_y), max_vertical_pixels), 0)
        )
    except Exception as e:
        print(f'**-topright_to_topleft:{e}')
    return a_point


def topleft_to_topright(a_point, max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y, unstitch_mode=False, offset_h=0, offset_v=0):
    """ Screen coordinates conversion Warning: reverses x & y """
    try:
        if unstitch_mode:
            a_reversed_point = (
                max(min(
                    round(adb_screen_max_y - (adb_screen_max_y-(a_point[0]*adb_screen_max_x/max_horizontal_pixels)+OFFSET_T_X) + offset_v), adb_screen_max_y), 0),
                max(min(
                    round(a_point[1]*adb_screen_max_x/max_vertical_pixels)-OFFSET_T_Y + offset_h, adb_screen_max_x), 0))
        else:
            a_reversed_point = (
                max(min(
                    round(a_point[1]*adb_screen_max_y/max_vertical_pixels-OFFSET_T_Y+a_point[1]*offset_v/max_vertical_pixels), adb_screen_max_y), 0),
                max(min(round(adb_screen_max_x-(a_point[0]*adb_screen_max_x/max_horizontal_pixels)+OFFSET_T_X+a_point[0]*offset_h/max_horizontal_pixels), adb_screen_max_x), 0))
    except Exception as e:
        print(f'**-topleft_to_topright:{e}')
    return a_reversed_point


def extract_metadata(source_fn):
    """ Extracts .note or .mark metadata """
    metadata = None
    # Retrieving base filename and extension
    basename, extension = os.path.splitext(source_fn)
    if extension.lower() in ['.note', '.mark']:
        try:
            with open(source_fn, 'rb') as f:
                metadata = sn.parse_metadata(f)
            # Saving the metadata
            metadata_json = metadata.to_json(indent=4)

            return metadata_json
        except Exception as e:
            print(f'Error in extract_metadata - type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            return {}


def get_pen_strokes_address_dict(note_fn, search_keyword=None):
    """ Returns a dictionary of totalpaths addresses for a given notebook named note_fn"""
    try:
        a_metadata = None
        fonts_page_number = -1
        result_dict = {}
        file_type = None
        if os.path.exists(note_fn) and note_fn[-5:].lower() == '.note':
            a_metadata = json.loads(extract_metadata(note_fn))
            if a_metadata is not None:
                # Retrieve the type of file
                if '__header__' in a_metadata:
                    a_header = a_metadata['__header__']
                    if 'APPLY_EQUIPMENT' in a_header:
                        file_type = a_header['APPLY_EQUIPMENT']
                if '__footer__' in a_metadata:
                    page_address_dict = a_metadata['__footer__']

                    if '__keywords__' in page_address_dict:
                        myfonts_keywords = page_address_dict['__keywords__']

                        if search_keyword:
                            for keyword in myfonts_keywords:
                                if keyword['KEYWORD'] == search_keyword:
                                    fonts_page_number = int(keyword['KEYWORDPAGE'])
                                    break

                    pdf_pages_list = [x[4:] for x in page_address_dict if x[:4] == 'PAGE']

                    if '__pages__' in a_metadata:
                        pages_list = a_metadata['__pages__']

                        pages_paths_list = [int(a_page['TOTALPATH']) for a_page in pages_list if 'TOTALPATH' in a_page.keys()]
                        pages_bitmap_list = [int(
                            a_page['__layers__'][0]['LAYERBITMAP']) for a_page in pages_list if len(a_page['__layers__']) > 0]

                        if fonts_page_number == -1:
                            for i in range(len(pdf_pages_list)):
                                result_dict[pdf_pages_list[i]] = {
                                    "totalpath": pages_paths_list[i],
                                    "bitmap": pages_bitmap_list[i]}
                        else:
                            result_dict[str(fonts_page_number)] = {
                                    "totalpath": pages_paths_list[fonts_page_number-1],
                                    "bitmap": pages_bitmap_list[fonts_page_number-1]}

        return result_dict, file_type, a_metadata
    except Exception as e:
        print()
        print(f'*** get_totalpath_address_dict: {e}')
        return None, None, None


def titles_and_text_from_notes(meta_data, note_fn):
    """ Retrieves useful information about titles and recognized text from notes"""
    with open(note_fn, 'rb') as a_note_file:
        binary_data = a_note_file.read()
    pages_dict = {}
    result = False
    if meta_data is not None:
        if '__pages__' in meta_data:
            pages_list = meta_data['__pages__']

            for page_index in range(len(pages_list)):
                a_page = pages_list[page_index]
                if 'RECOGNTEXT' in a_page:
                    a_page_str_rec = a_page['RECOGNTEXT']
                    if a_page_str_rec != "0":
                        a_page_rec_pos = int(a_page_str_rec)
                        completed, size_rec = read_endian_int_at_position(binary_data, a_page_rec_pos)
                        if completed:
                            try:
                                begin_read = a_page_rec_pos + 4
                                end_read = begin_read + size_rec
                                byte_string = bytes(binary_data[begin_read:end_read])
                                encoded_json = base64.b64decode(byte_string).decode('utf-8')
                                rec_text = json.loads(encoded_json)
                                pages_dict[str(page_index)] = rec_text
                            except Exception as e:
                                print()
                                print(f'*** Error retrieving rec text: {e} -- page:{page_index}')
                                print(encoded_json)
                                exit(0)

    result = True
    return result, pages_dict


def get_pen_strokes_dict(note_fn, search_keyword=None):
    """ Returns a dictionary of totalpaths for a given notebook named note_fn.
        the root keys are the page numbers"""
    try:
        offset_x = 0
        offset_y = 0
        pen_strokes_address_dict, file_type, meta_data = get_pen_strokes_address_dict(note_fn, search_keyword=search_keyword)

        if pen_strokes_address_dict is not None:

            with open(note_fn, 'rb') as a_note_file:
                note_file_binaries = a_note_file.read()

            if is_n6_file(note_file_binaries):
                print('N6 file type')
                detected_series = 'N6'
            else:
                print('N5 file type')
                detected_series = 'N5'

            unique_used_list = []

            last_good_address = 0

            for a_page in pen_strokes_address_dict.keys():
                page_pen_strokes_dict = pen_strokes_address_dict[a_page]

                page_totalpath_address = page_pen_strokes_dict['totalpath']

                if page_totalpath_address != 0:
                    _, page_totalpath_size = read_endian_int_at_position(note_file_binaries, page_totalpath_address)
                    _, page_totalpath_strokes_nb = read_endian_int_at_position(note_file_binaries, page_totalpath_address + 4)
                    page_pen_strokes_dict['size'] = page_totalpath_size
                    page_pen_strokes_dict['strokes_nb'] = page_totalpath_strokes_nb

                    pen_strokes_list = []
                    a_position = page_totalpath_address + 8
                    for pen_stroke_index in range(page_totalpath_strokes_nb):

                        _, pen_stroke_size = read_endian_int_at_position(note_file_binaries, a_position)

                        _, pen_type = read_endian_int_at_position(note_file_binaries, a_position + 4, num_bytes=1)
                        # We are skipping the below because it pertains to headings and links (not pen strokes)
                        if pen_type == 0:
                            print(f'**- skipped at address: {a_position}')
                            a_position += pen_stroke_size + 4
                            continue
                        _, pen_color = read_endian_int_at_position(note_file_binaries, a_position + 8, num_bytes=1)
                        _, pen_weight = read_endian_int_at_position(note_file_binaries, a_position + 12, num_bytes=2)
                        _, min_contours_x = read_endian_int_at_position(note_file_binaries, a_position + 104)
                        _, min_contours_y = read_endian_int_at_position(note_file_binaries, a_position + 108)
                        _, avg_contours_x = read_endian_int_at_position(note_file_binaries, a_position + 112)
                        _, avg_contours_y = read_endian_int_at_position(note_file_binaries, a_position + 116)
                        _, max_contours_x = read_endian_int_at_position(note_file_binaries, a_position + 120)
                        _, max_contours_y = read_endian_int_at_position(note_file_binaries, a_position + 124)
                        _, vector_size = read_endian_int_at_position(note_file_binaries, a_position + 216)
                        if vector_size == 0:
                            print(f'**- Zero size vector skipped at address: {a_position}')
                            a_position += pen_stroke_size + 4
                            continue

                        vector_points = []
                        for index_point in range(vector_size):
                            _, point_y = read_endian_int_at_position(note_file_binaries, a_position + 220 + index_point * 8)
                            _, point_x = read_endian_int_at_position(note_file_binaries, a_position + 220 + index_point * 8 + 4)
                            vector_points.append([point_y, point_x])

                        pressure_points = []
                        for index_pressure in range(vector_size):
                            _, ppoint = read_endian_int_at_position(note_file_binaries, a_position + 224 + vector_size * 8 + index_pressure*2, num_bytes=2)
                            pressure_points.append(ppoint)

                        # Skip the headear of the unique vector to read the first unique number
                        _, unique_number = read_endian_int_at_position(note_file_binaries, a_position + 228 + vector_size * 8 + vector_size*2, num_bytes=4)

                        # Build the unique vector. TODO: Probably better to just read it
                        unique_used_list = [unique_number]*vector_size

                        # Then there is a vector of ones, vector_size*1 byte

                        # position is now:  a_position + 228 + vector_size * 8 + vector_size*2 + 4 + vector_size*4 + vector_size*1
                        #                   a_position + 232 + vector_size * 15
                        #
                        # Not sure if I had corrupted files, or if version changed, but some of my files have a sequence 0f 12 zero bytes
                        # following the "vector of 1"
                        # Let's read the sequence length
                        # sequence_length = read_zero_sequence(note_file_binaries, a_position + 232 + vector_size * 15)
                        sequence_length = 16  # Ignoring read_zero_sequence

                        #                 > We add sequence_length bytes of zeros
                        #                   a_position + 232 + sequence_length + vector_size * 15
                        #                 > We add 4 bytes for the pen stroke number
                        #                   a_position + 236 + sequence_length + vector_size * 15
                        #                 > We add 54 bytes
                        #                   a_position + 290 + sequence_length + vector_size * 15

                        _, contours_number = read_endian_int_at_position(note_file_binaries, a_position + 290 + sequence_length + vector_size * 15, num_bytes=4)

                        # position is now:  a_position + 310 + vector_size * 15
                        contour_processed_nb = 0
                        contours_dict = {}
                        contour_count = 0

                        # contour_problem = False
                        # print(f'----contours_number: {contours_number}')
                        if contours_number > 1000:
                            print(f'**- TOO HIGH contours_number: {contours_number} for page: {a_page} - Pen stroke #: {pen_stroke_index} pen_type: {pen_type}')
                            print(f'> page_totalpath_address: {page_totalpath_address}')
                            print(f'> last good trail address: {last_good_address}')
                            print(f'> Failing Trail Address: {a_position}')
                            print(f'> Failing contour Address: {a_position + 290 + sequence_length + vector_size * 15}')
                            print(f'  Pen_stroke_size:{pen_stroke_size} at: {a_position}')
                            print(f' Vector_size:{vector_size} at: {a_position + 216}')
                            return None, None, None

                        for contour_index in range(contours_number):
                            contour_count_fails = 0
                            _, contour_size = read_endian_int_at_position(note_file_binaries, a_position + 310 + vector_size * 15 + contour_processed_nb + contour_count, num_bytes=4)
                            contours_list = []

                            last_contour_point = None

                            for contour_element_idx in range(contour_size):
                                contour_x = decimal_ieee754_from_binary(note_file_binaries, a_position + 314 + vector_size * 15 + contour_element_idx*8 + contour_count, offset_x)
                                contour_y = decimal_ieee754_from_binary(note_file_binaries, a_position + 318 + vector_size * 15 + contour_element_idx*8 + contour_count, offset_y)
                                if contour_x and contour_y:
                                    last_contour_point = [contour_x, contour_y]
                                    contours_list.append(last_contour_point)
                                else:
                                    if last_contour_point:
                                        contours_list.append(last_contour_point)
                                    else:
                                        contour_count_fails += 1

                            contours_dict[str(contour_index)] = list(contours_list)
                            # contour_count += (contour_size-contour_count_fails)*8 + 4
                            contour_count += (contour_size)*8 + 4

                        remaining_bytes_start = a_position + 318 + vector_size * 15 + contour_count-4
                        remaining_bytes_end = a_position + pen_stroke_size + 4
                        remaining_bytes = note_file_binaries[remaining_bytes_start:remaining_bytes_end]

                        pen_strokes_list.append({
                            'address': a_position, 'size': pen_stroke_size,
                            'type': pen_type, 'color': pen_color, 'weight': pen_weight,
                            'min_c_x': min_contours_x, 'min_c_y': min_contours_y, 'avg_c_x': avg_contours_x, 'avg_c_y': avg_contours_y,
                            'max_c_x': max_contours_x, 'max_c_y': max_contours_y, 'vector_size': vector_size, 'vector_points': vector_points,
                            'vector_pressure': pressure_points, 'vector_unique': unique_used_list, 'vector_one': [1]*vector_size,
                            'contours': contours_dict, 'r_bytes': remaining_bytes.hex().upper()})

                        last_good_address = a_position
                        a_position += pen_stroke_size + 4
                    page_pen_strokes_dict['strokes'] = pen_strokes_list
                else:
                    page_pen_strokes_dict['strokes'] = []

        return pen_strokes_address_dict, file_type, meta_data, detected_series
    except Exception as e:
        print()
        print(f'*** get_pen_strokes_dict: {e}')
        return None, None, None, ''


def draw_bitmap_from_vectors(vector_dict, image_size=(500, 500), image_crop=None, background_color=(255, 255, 255), line_width=3):
    """
    Draws a bitmap from a list of vectors of points.
    Args:
    - vectors: List of vectors, where each vector is a list of (x, y) points.
    - image_size: Tuple specifying the size of the output image (width, height).
    - background_color: Tuple representing the RGB color of the background.
    - line_color: Tuple representing the RGB color of the drawn lines.
    - line_width: Width of the lines.
    Returns:
    - An Image object containing the drawn bitmap.
    """
    try:
        # Create a blank image with the specified background color
        image = Image.open(BLANK_TEMPLATE_PICTURE2)

        # image = Image.new("RGB", image_size, background_color)
        image = image.convert('L')
        draw = ImageDraw.Draw(image)

        for a_color_str, vectors in vector_dict.items():

            a_color = int(a_color_str)
            # line_color = (a_color, a_color, a_color)
            line_color = a_color

            for vector in vectors:
                # Iterate over each vector (path of points) and draw lines between the points

                if len(vector) > 1:  # Only draw if there are at least two points
                    draw.line(vector, fill=line_color, width=line_width)

        if image_crop is not None:
            image = image.crop((image_crop[0], image_crop[1], image_crop[0]+image_crop[2], image_crop[1]+image_crop[3]))

    except Exception as e:
        print(f'*** draw_bitmap_from_vectors: {e}')
        return None
    return image


def rle_encode_img(image):
    """ Ephemeral bitmap images are encoded in the .note and .mark files
        using a 'simple' RLE algorithm.
        Encodes image data in a run-length algorithm compatible with the
        supernote decode. This assumes that palette is none and blank is none
        TODO: comment all steps
    """
    encoded = False
    try:
        segments_list = []
        result = bytearray()
        # Convert the image to grayscale mode 'L'
        try:
            gray_image = image.convert('L')
        except Exception as e:
            print(f'*** Convert image: {e}')
            return image, True

        # Convert the grayscale image to a NumPy array
        img_array = np.array(gray_image)
        # Flatten the array to 1D
        pixels = img_array.flatten()
        # Detect where the value changes
        changes = np.diff(pixels) != 0
        change_indices = np.where(changes)[0] + 1
        # Include the start and end of the array
        segment_starts = np.concatenate(([0], change_indices))
        segment_ends = np.concatenate((change_indices, [len(pixels)]))

        large_threshold = int(str(SPECIAL_LENGTH))
        zip_data = zip(segment_starts, segment_ends)

        for start, end in zip_data:
            length_s = end - start
            color_s = pixels[start]

            segments_list.append((start, color_s, length_s))
            if color_s == 255:
                color_s = COLORCODE_BACKGROUND  # 62
            elif color_s == 0:
                color_s = 0x61
            elif color_s == 254:
                color_s = COLORCODE_WHITE  # 65
            elif color_s == 201:
                color_s = 0xCA
            elif color_s == 157:
                color_s = 0x9E  # 63
            else:
                color_s = 0xb7

            if length_s > 127:
                q_128, r_128 = divmod(length_s, 128)
                if r_128 == 0 and (128+q_128-1 < 255):
                    result.extend([color_s] + list(bytes.fromhex(format(128+q_128-1, '02x'))))
                else:
                    quotient, remainder = divmod(length_s, large_threshold)
                    result.extend([color_s, SPECIAL_LENGTH_MARKER]*quotient)
                    first_part = int((remainder-1)/128) + 127
                    second_part = remainder-1-((first_part % 127)*128)
                    try:
                        b_1 = bytes.fromhex(format(first_part, '02x'))
                        b_2 = bytes.fromhex(format(second_part, '02x'))
                        if first_part == 127:
                            result.extend([color_s] + list(b_2))
                        else:
                            result.extend([color_s] + list(b_1) + [color_s] + list(b_2))
                    except Exception as e:
                        # TODO: Find a flawless RLE encoder
                        # This is a problem in the encoder, (remainder == 0 creates negative "second_part)
                        # but I don't wan to display the error on the terminal and
                        # I don't want to change my flake rules so leaning "e"
                        e
            else:
                try:
                    result.extend([color_s]+list(bytes.fromhex(format(length_s - 1, '02x'))))
                except Exception as e:
                    print(f'*** Error in RLE: {e}')

        encoded = True
    except Exception as e:
        print()
        print(f'*** Error in rle_encode_img: {e}')
        print(f'- 1:{first_part} 2:{second_part} r_128:{r_128} q:{128+q_128-1} length_s:{length_s - 1} - r: {128+q_128-1}')
        return None, encoded

    return result, encoded


def strokes_at_point(strokes_list, x, y, row_height=90, scratio=1, get_xt_point=False, new_weight=None, new_color=None, series=NOTEBOOK_DEVICE):
    """ Returns a modified strokes list for coordinates x,y (ref MAX_HORIZONTAL_PIXELS/MAX_VERTICAL_PIXELS)
        The function first checks the relative delta_x and delta_y for all the
        strokes in the list. It then applies the x/y coordinates to that point
        and shifts the other strokes by their delta.
        We do not check if the shifting movs the strokes out of the screen boundaries

        strokes_list is a list of dictionaries"""
    try:

        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)

        min_x = min([x['min_c_x'] for x in strokes_list])
        max_x = max([x['max_c_x'] for x in strokes_list])
        min_y = min([x['min_c_y'] for x in strokes_list])
        max_y = max([x['max_c_y'] for x in strokes_list])

        stroke_width = max_x - min_x
        stroke_height = max_y - min_y

        y_adjustment = max(0, row_height - stroke_height)
        delta_x = x - min_x

        delta_y = y - min_y + y_adjustment
        delta_x_mu = -1 * round(delta_x * adb_screen_max_x / max_horizontal_pixels)

        delta_y_mu = round(delta_y * adb_screen_max_y / max_vertical_pixels)

        scratio_x = round(adb_screen_max_x*(1-scratio)*(1-MARGIN_LEFT_SIZE))

        scratio_pix = round(max_horizontal_pixels*(1-scratio)*(1-MARGIN_LEFT_SIZE))

        # Create a deep copy of the list
        mod_strokes_list = copy.deepcopy(strokes_list)

        # Parse the list

        for ps_dict in mod_strokes_list:

            new_min_x = round(scratio*(ps_dict['min_c_x'] + delta_x)) + scratio_pix
            new_max_x = round(scratio*(ps_dict['max_c_x'] + delta_x)) + scratio_pix
            new_min_y = round(scratio*(ps_dict['min_c_y'] + delta_y))
            new_max_y = round(scratio*(ps_dict['max_c_y'] + delta_y))

            if new_weight:
                ps_dict['weight'] = new_weight

            if new_color:
                ps_dict['color'] = new_color

            ps_dict['min_c_x'] = max(min(new_min_x, max_horizontal_pixels), 0)
            ps_dict['min_c_y'] = max(min(new_min_y, max_vertical_pixels), 0)
            ps_dict['max_c_x'] = max(min(new_max_x, max_horizontal_pixels), 0)
            ps_dict['max_c_y'] = max(min(new_max_y, max_vertical_pixels), 0)
            ps_dict['avg_c_x'] = max(min(round(scratio*(ps_dict['avg_c_x'] + delta_x)) + scratio_x, max_horizontal_pixels), 0)
            ps_dict['avg_c_y'] = max(min(round(scratio*(ps_dict['avg_c_y'] + delta_y)), max_vertical_pixels), 0)  # TODO: Based on 1 example. this was unchanged. WHY? confirm
            xt_vector = [
                [
                    max(min(round(scratio*(x[0] + delta_y_mu)), adb_screen_max_y), 0),
                    max(min(round(scratio*(x[1] + delta_x_mu) + scratio_x), adb_screen_max_x), 0)] for x in ps_dict['vector_points']]
            ps_dict['vector_points'] = xt_vector

            if get_xt_point:
                xt_x_list = [x[1] for x in xt_vector]
                xt_y_list = [x[0] for x in xt_vector]
                xt_rect = [min(xt_x_list), min(xt_y_list), max(xt_x_list), max(xt_y_list)]
            else:
                xt_rect = []

            mod_contours = {}
            for contour_nb, contour_value in ps_dict['contours'].items():
                new_list = [
                    [
                        max(min(round(scratio*(x[0] + delta_x) + scratio_pix), max_horizontal_pixels), 0),
                        max(min(round(scratio*(x[1] + delta_y)), max_vertical_pixels), 0)] for x in contour_value]
                mod_contours[contour_nb] = new_list
            ps_dict['contours'] = mod_contours

        return mod_strokes_list, stroke_width, stroke_height, xt_rect
    except Exception as e:
        print()
        print(f'*** strokes_at_point: {e}')
        return None, 0, 0, []


def get_pen_stokes_list_from_table(series, font_name=FONT_NAME):
    """ Gets the list of pen strokes.
        Parameters:
        - asciiset_fn: A Notebook containing an fonts in an ASCII table format
                        See char_from_ascii_tps for default format of that table
        It is preferable to store this in one variable as Mac with 8GB reaches some limits"""
    try:
        asciiset_fn = os.path.join(CONFIG, f'fonts_{series.lower()}.note')
        print(f'   > Getting penstrokes for font: {font_name} using {asciiset_fn}')
        print()
        ascii_type = None
        # Load the dictionary of pen strokes contained in the fonts note
        ascii_pen_strokes_dict, ascii_type, _, _ = get_pen_strokes_dict(asciiset_fn, search_keyword=font_name)

        # Extract its list of pen strokes (we assume the ascii table fits in page 1)
        for akey, avalue in ascii_pen_strokes_dict.items():
            return avalue['strokes'], ascii_type

    except Exception as e:
        print(f'*** get_pen_stokes_list_from_table: {e}')
        return [], None


def intersection_area(rect1, rect2, threshold=0.95):
    """ Checks if 2 rects are intersecting above a threshold """
    # Unpack the coordinates of the rectangles

    x0, y0, x1, y1 = rect1
    a0, b0, a1, b1 = rect2

    rect1_area = (x1-x0)*(y1-y0)
    rect2_area = (a1-a0)*(b1-b0)
    rect_area = min(rect1_area, rect2_area)

    # Find the coordinates of the intersection rectangle
    x_left = max(x0, a0)
    y_bottom = max(y0, b0)
    x_right = min(x1, a1)
    y_top = min(y1, b1)

    # Check if there is no overlap
    if x_left > x_right or y_bottom > y_top:
        return False

    # Calculate the width and height of the intersection rectangle
    width = x_right - x_left
    height = y_top - y_bottom
    coverage = width * height / rect_area
    if coverage > threshold:
        return True
    return False


def sn_bytes_header(version='20230015', type='note'):
    """ Returns the file signature for the binary header """
    signature = f'{type}SN_FILE_VER_{version}'
    return signature.encode('utf-8')


def sn_bytes_1_header_json(
    original_style='style_white', original_md5=0, templates_dict={}, file_type='NOTE',
        apply_equipment=NOTEBOOK_DEVICE, final_op_page=1, reco_lang=FILE_RECOGN_LANGUAGE,
        style_usage_type=0):
    """ Generates a file_id and returns it, alongside with the very first json, prefixed by its size"""
    # TODO: apply_equipment=NOTEBOOK_DEVICE N5 doesn't work on Manta. Perhaps need to check the adb max values, so hardcoding to N6 for now
    # print(f'-----apply_equipment:{apply_equipment} ADB_SCREEN_MAX_X:{ADB_SCREEN_MAX_X}  ADB_SCREEN_MAX_y:{ADB_SCREEN_MAX_Y}')

    # # # for a_page_t, a_value_t in templates_dict.items():
    # # #     an_md5 = a_value_t['md5']
    # # #     an_md5_ = an_md5.split('_')
    # # #     original_style = f'user_{an_md5_[0]}'
    # # #     original_md5 = an_md5
    # # #     break
    if reco_lang == 'none':
        text_reco = 0
    else:
        text_reco = 1

    file_id = f'F{datetime_identifier()}'
    part_1 = f'<MODULE_LABEL:SNFILE_FEATURE><FILE_TYPE:{file_type}><APPLY_EQUIPMENT:{apply_equipment}><FINALOPERATION_PAGE:{final_op_page}>'

    # part_2 = f'<FINALOPERATION_LAYER:1><ORIGINAL_STYLE:{original_style}><ORIGINAL_STYLEMD5:{original_md5}><DEVICE_DPI:0><SOFT_DPI:0><FILE_PARSE_TYPE:0><RATTA_ETMD:0><APP_VERSION:0>'

    part_2 = '<FINALOPERATION_LAYER:1><DEVICE_DPI:0><SOFT_DPI:0><FILE_PARSE_TYPE:0><RATTA_ETMD:0><APP_VERSION:0>'

    part_3 = f'<FILE_ID:{file_id}><FILE_RECOGN_TYPE:{text_reco}><FILE_RECOGN_LANGUAGE:{reco_lang}>'

    if len(templates_dict.keys()) > 0:
        used_styles_list = []
        part_4 = ''
        for a_page in templates_dict:
            template_page_dict = templates_dict[a_page]
            md5 = template_page_dict['md5']
            if md5 not in used_styles_list:
                used_styles_list.append(md5)
                # style_p1_name = f'user_pdf_xyz_{len(used_styles_list)}'
                style_p1_name = f'user_{md5}'
                pdfstylelist_name = f'{style_p1_name}_{md5}'
                pdfstylelist_name_encoded = base64.b64encode(pdfstylelist_name.encode('utf-8'))
                template_page_dict['pdfstylelist'] = {
                    'name': pdfstylelist_name,
                    'encoded': pdfstylelist_name_encoded}
                # part_4 += f'<PDFSTYLE:{style_p1_name}>'
                # part_4 += f'<PDFSTYLEMD5:{md5}>'
                part_4 += '<PDFSTYLE:none>'
                part_4 += '<PDFSTYLEMD5:0>'
        part_4 += '<STYLEUSAGETYPE:1>'
    else:
        part_4 = '<PDFSTYLE:none><PDFSTYLEMD5:0><STYLEUSAGETYPE:0>'
    part_5 = '<HIGHLIGHTINFO:0><HORIZONTAL_CHECK:0><IS_OLD_APPLY_EQUIPMENT:1><ANTIALIASING_CONVERT:2>'
    json_ = f'{part_1}{part_2}{part_3}{part_4}{part_5}'.encode('utf-8')
    # Computing firs json's size
    json_s = int_to_little_endian_bytes(len(json_))
    result = json_s + json_

    return file_id, result, templates_dict


def blank_encoded_image(device):
    if device in ['N5']:
        background_image_block = bytes.fromhex((
            '5802000062FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'))
    else:
        background_image_block = bytes.fromhex((
            '4401000062FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF'
            '62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62FF62B4623F'))
    return background_image_block


def pen_strokes_dict_to_bytes(pen_strokes_dict, num_bytes=4, byteorder='little', series=NOTEBOOK_DEVICE):
    """ Generates penstroke bytes from dictionary """
    try:
        bytes_dict = {}
        offset_x = 0
        offset_y = 0
        print()
        print('  > Generating binaries')
        for page_str, page_pen_strokes_dict in pen_strokes_dict.items():
            print(f'   - page {page_str}')
            page_in_bytes = int(page_str).to_bytes(2, byteorder=byteorder)
            page_tpath_strokes_nb = page_pen_strokes_dict['strokes_nb'].to_bytes(num_bytes, byteorder=byteorder)
            page_stroke_dicts = page_pen_strokes_dict['strokes']

            page_stroke_nb = 1

            total_stroke_bytes = page_tpath_strokes_nb

            for a_stroke_dict in page_stroke_dicts:

                if a_stroke_dict == {}:
                    print('*** Fatal error: empty dict in pen_strokes_dict_to_bytes, exiting')
                    exit(1)

                a_stroke_bytes = None

                stroke_pen_type = a_stroke_dict['type'].to_bytes(1, byteorder=byteorder)

                if a_stroke_bytes is None:
                    a_stroke_bytes = stroke_pen_type
                else:
                    a_stroke_bytes += stroke_pen_type

                a_stroke_bytes += bytes.fromhex('000000')

                stroke_color = signed_to_bytes(a_stroke_dict['color'], 1, byteorder=byteorder)
                a_stroke_bytes += stroke_color
                a_stroke_bytes += bytes.fromhex('000000')
                stroke_weight = signed_to_bytes(a_stroke_dict['weight'], 2, byteorder=byteorder)

                a_stroke_bytes += stroke_weight
                a_stroke_bytes += bytes.fromhex('00000A0000000000000020000000FFFFFFFF')
                a_stroke_bytes += page_in_bytes
                a_stroke_bytes += bytes.fromhex('0000000000000000000088130000000000006F746865727300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')

                a_stroke_min_c_x = signed_to_bytes(a_stroke_dict['min_c_x'], 4, byteorder=byteorder)
                a_stroke_bytes += a_stroke_min_c_x
                a_stroke_min_c_y = signed_to_bytes(a_stroke_dict['min_c_y'], 4, byteorder=byteorder)
                a_stroke_bytes += a_stroke_min_c_y
                a_stroke_avg_c_x = signed_to_bytes(a_stroke_dict['avg_c_x'], 4, byteorder=byteorder)
                a_stroke_bytes += a_stroke_avg_c_x
                a_stroke_avg_c_y = signed_to_bytes(a_stroke_dict['avg_c_y'], 4, byteorder=byteorder)

                a_stroke_bytes += a_stroke_avg_c_y
                a_stroke_max_c_x = signed_to_bytes(a_stroke_dict['max_c_x'], 4, byteorder=byteorder)
                a_stroke_bytes += a_stroke_max_c_x
                a_stroke_max_c_y = signed_to_bytes(a_stroke_dict['max_c_y'], 4, byteorder=byteorder)
                a_stroke_bytes += a_stroke_max_c_y

                if series in ['N5']:
                    somehexbytes = '0200000080540000603F000073757065724E6F74654E6F746500000000000000000000000000000000000000000000'
                else:
                    somehexbytes = '02000000CB3D0000582E000073757065724E6F74654E6F746500000000000000000000000000000000000000000000'

                somehexbytes += '0000000000000000000000000000000000010000000000000000000000000000000000000000000000'

                a_stroke_bytes += bytes.fromhex(somehexbytes)

                a_stroke_vector_points = a_stroke_dict['vector_points']

                a_stroke_vector_points_length = len(a_stroke_vector_points)
                a_stroke_bytes += a_stroke_vector_points_length.to_bytes(4, byteorder=byteorder)
                for x in a_stroke_vector_points:
                    a_stroke_bytes += signed_to_bytes(x[0], num_bytes, byteorder=byteorder)
                    a_stroke_bytes += signed_to_bytes(x[1], num_bytes, byteorder=byteorder)
                a_stroke_vector_pressure = a_stroke_dict['vector_pressure']

                a_stroke_bytes += a_stroke_vector_points_length.to_bytes(4, byteorder=byteorder)
                vector_pressure_bytes = [
                    signed_to_bytes(x, 2, byteorder=byteorder) for x in a_stroke_vector_pressure]
                a_stroke_bytes += b''.join(vector_pressure_bytes)
                a_stroke_vector_unique = a_stroke_dict['vector_unique']
                a_stroke_bytes += signed_to_bytes(a_stroke_vector_points_length, 4, byteorder=byteorder)
                vector_unique_bytes = [
                    signed_to_bytes(x, 4, byteorder=byteorder) for x in a_stroke_vector_unique]
                a_stroke_bytes += b''.join(vector_unique_bytes)
                a_stroke_vector_one = a_stroke_dict['vector_one']
                a_stroke_bytes += signed_to_bytes(a_stroke_vector_points_length, 4, byteorder=byteorder)
                vector_one_bytes = [
                    signed_to_bytes(x, 1, byteorder=byteorder) for x in a_stroke_vector_one]
                a_stroke_bytes += b''.join(vector_one_bytes)
                a_stroke_bytes += bytes.fromhex('00000000000000000000000000000000')
                a_stroke_bytes += page_stroke_nb.to_bytes(4, byteorder=byteorder)  # TODO: Validate 241029 correction The number of strokes is likely at least 4 bytes
                a_stroke_bytes += bytes.fromhex('000000000000000000000000000000000100000001000000000000000000000001000000010000000000000000000000000000000000')
                a_stroke_contours = a_stroke_dict['contours']

                a_stroke_contours_len = len(a_stroke_contours.keys())

                a_stroke_bytes += a_stroke_contours_len.to_bytes(4, byteorder=byteorder)

                for contour_nb, contour_vector in a_stroke_contours.items():

                    a_c_v = None
                    len_a_c_v = 0
                    for a_point in contour_vector:
                        d_x = decimal_to_custom_ieee754(a_point[0], offset=offset_x)
                        d_y = decimal_to_custom_ieee754(a_point[1], offset=offset_y)
                        if d_x and d_y:
                            if a_c_v is None:
                                a_c_v = d_x
                            else:
                                a_c_v += d_x
                            a_c_v += d_y
                            len_a_c_v += 1

                    if a_c_v:
                        a_stroke_bytes += len_a_c_v.to_bytes(4, byteorder=byteorder)
                        a_stroke_bytes += a_c_v

                a_stroke_bytes += a_stroke_contours_len.to_bytes(4, byteorder=byteorder)

                a_stroke_bytes += bytes.fromhex(a_stroke_dict['r_bytes'])

                page_stroke_nb += 1
                len_page_strokes_bytes = len(a_stroke_bytes).to_bytes(4, byteorder=byteorder)
                a_stroke_bytes = len_page_strokes_bytes + a_stroke_bytes
                if total_stroke_bytes is None:
                    total_stroke_bytes = a_stroke_bytes
                else:
                    total_stroke_bytes += a_stroke_bytes

            bytes_dict[page_str] = total_stroke_bytes
        return bytes_dict
    except Exception as e:
        print()
        print(f'*** pen_strokes_dict_to_bytes: {e}')


def generate_ephemeral_images(
    pen_strokes_dict,
        image_crop=None, background_color=(254, 254, 254, 254), line_width=3, series=NOTEBOOK_DEVICE):
    """
    From a pen_strokes_dict, modifies a binary_data and returns it.
    if the current ephemeral image is larger than the generated image from the pen strokes
    """
    try:
        full_image_dict = {}
        a_stroke_color = None
        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)

        print()
        print('  > Generating images')
        # Parse the dictionary of totalpath objects
        for apage, avalue in pen_strokes_dict.items():

            full_image_block = None

            # Create a blank image with the specified background color
            if series in ['N5']:
                image = Image.open(BLANK_TEMPLATE_PICTURE2)
            else:
                image = Image.open(BLANK_TEMPLATE_PICTURE)

            draw = ImageDraw.Draw(image)

            # Retrieve the list of all pen stroke objects on page "apage"
            a_list_strokes = avalue['strokes']

            # Parse each stroke on that page and draw it
            for a_stroke in a_list_strokes:

                a_stroke_color = a_stroke['color']

                # line_color = (a_stroke_color, a_stroke_color, a_stroke_color)
                line_color = a_stroke_color
                a_stroke_vector_points = a_stroke['vector_points']
                normalized_points = [
                    topright_to_topleft(
                        x, max_horizontal_pixels, max_vertical_pixels,
                        adb_screen_max_x, adb_screen_max_y) for x in a_stroke_vector_points]
                a_stroke_vector_size = a_stroke['vector_size']

                if a_stroke_vector_size > 1:  # Only draw if there are at least two points
                    draw.line(normalized_points, fill=line_color, width=line_width)
                else:
                    print(f'*** too short vector of size: {a_stroke_vector_size}')

            # Crop if necessary
            if image_crop is not None:
                image = image.crop((image_crop[0], image_crop[1], image_crop[0]+image_crop[2], image_crop[1]+image_crop[3]))

            # Encode the image
            image = image.convert('L')
            # image.show()

            rleimage, encoded = rle_encode_img(image)

            if encoded:
                # Now check if the size fits in the current bitmap space
                new_image_size = len(rleimage)

                # Encode the image size
                image_size_bytes = int_to_little_endian_bytes(new_image_size)

                # Append the size to create the full image block
                full_image_block = image_size_bytes + rleimage

                print(f'   - page {apage}: {len(full_image_block)} bytes image')

            full_image_dict[apage] = full_image_block

    except Exception as e:
        print(f'*** generate_ephemeral_images: {e}')
        return None

    return full_image_dict


def sn_bytes_layer_json(bitmap_address, file_type='NOTE', protocol='RATTA_RLE', name='MAINLAYER', layerpath=0, vectorgraph=0, recogn=0):
    """ Generates json for a layer and returns it, prefixed by its size"""
    a_json = f'<LAYERTYPE:{file_type}><LAYERPROTOCOL:{protocol}><LAYERNAME:{name}><LAYERPATH:{layerpath}>'
    a_json += f'<LAYERBITMAP:{bitmap_address}><LAYERVECTORGRAPH:{vectorgraph}><LAYERRECOGN:{recogn}>'
    a_json_ = a_json.encode('utf-8')

    # Computing json's size
    json_s = int_to_little_endian_bytes(len(a_json_))
    result = json_s + a_json_
    return result


def build_titles_image_n_details(titles_dict, current_bytes):
    """ Generates successive blocks of title bitmap blocks,
        followed by title details block"""

    titles_recap = ''

    try:

        for a_page, a_page_titles in titles_dict.items():

            for a_page_title in a_page_titles:

                starting_position = len(current_bytes)
                title_bitmap_rect = a_page_title[0]
                title_bitmap_hex = a_page_title[1]
                title_bitmap_bytes = bytes.fromhex(title_bitmap_hex)

                current_bytes += title_bitmap_bytes

                detail_location = len(current_bytes)
                x, y, dx, dy = title_bitmap_rect
                title_detail_str = f'<TITLESEQNO:0><TITLELEVEL:1><TITLERECT:{x},{y},{dx},{dy}><TITLERECTORI:{x},{y},{dx},{dy}>'
                title_detail_str += f'<TITLEBITMAP:{starting_position}><TITLEPROTOCOL:RATTA_RLE><TITLESTYLE:1201000>'
                title_detail_str_ = title_detail_str.encode('utf-8')
                title_detail_str_size = len(title_detail_str_)
                title_detail_str_size_bytes = int_to_little_endian_bytes(title_detail_str_size)

                current_bytes += title_detail_str_size_bytes + title_detail_str_

                titles_recap += f'<TITLE_{a_page.zfill(4)}{str(y).zfill(4)}{str(x).zfill(4)}:{detail_location}>'

    except Exception as e:
        print(f'*** build_titles_image_n_details: {e}')

    return current_bytes, titles_recap


def generate_link_json(sn_page_nb, pdf_path, pdf_page_nb, link_rect, binary):
    """ Appends link json to a binary and return updated binary and location """
    try:
        # Insertion position
        link_bitmap = len(binary)

        # Inserts a default link bitmap of reasonable size
        # binary += bytes.fromhex('0800000062FF62FF6289625F')
        binary += bytes.fromhex('0800000062FF62FF62D06226')

        link_json_address = len(binary)

        # Supernote path to pdf
        sn_file_path = str(base64.b64encode(f'/storage/emulated/0{pdf_path}'.encode('utf-8')))[2:-1]

        # Create the name of the link
        x0, y0, dx, dy = link_rect
        link_name = f'LINKO_{str(sn_page_nb).zfill(4)}{str(y0).zfill(4)}{str(x0).zfill(4)}{str(dy).zfill(4)}{str(dx).zfill(4)}'

        # Create the JSON
        link_json = f'<LINKTYPE:2><LINKINOUT:0><LINKBITMAP:{link_bitmap}><LINKSTYLE:20000000>'
        link_json += '<LINKTIMESTAMP:' + str(datetime.now().strftime("%Y%m%d%H%M%S%f")) + '>'
        link_json += f'<LINKRECT:{x0},{y0},{dx},{dy}><LINKRECTORI:{x0},{y0},{dx},{dy}>'
        link_json += f'<LINKPROTOCAL:RATTA_RLE><LINKFILE:{sn_file_path}><LINKFILEID:none><PAGEID:none><OBJPAGE:{pdf_page_nb}>'
        link_json += '<FONTSIZE:0.000000><FONTPATH:none><FULLTEXT:none><SHOWTEXT:none>'
        encoded_link_json = link_json.encode('utf-8')
        encoded_link_json_size = int_to_little_endian_bytes(len(encoded_link_json))
        link_json_block = encoded_link_json_size + encoded_link_json
        binary += link_json_block
    except Exception as e:
        print(f'*** generate_link_json: {e}')
        return None, None, binary
    return link_name, link_json_address, binary


def sn_bytes_page(
    main_layer_address, background_layer_address, totalpath_address,
        apply_equipment=NOTEBOOK_DEVICE, page_style_name=None,
        page_style_md5=None, rec_status=0, rec_address=0, recf_status=0, rec_page_type=0):
    """ Generates json for a page and returns it, prefixed by its size"""
    page_id = f'P{datetime_identifier()}'

    if page_style_name is None:
        if apply_equipment == 'N5':
            a_json = '<PAGESTYLE:style_white_a5x2>'
        else:
            a_json = '<PAGESTYLE:style_white>'
    else:
        a_json = f'<PAGESTYLE:{page_style_name}>'

    if page_style_md5 is None:
        a_json += '<PAGESTYLEMD5:0>'
    else:
        a_json += f'<PAGESTYLEMD5:{page_style_md5}>'

    a_json += '<LAYERINFO:[{"layerId"#3,"name"#"Layer 3","isBackgroundLayer"#false,"isAllowAdd"#false,'
    a_json += '"isCurrentLayer"#false,"isVisible"#true,"isDeleted"#true,"isAllowUp"#false,"isAllowDown"#false},{"layerId"#2,"name"#"Layer 2",'
    a_json += '"isBackgroundLayer"#false,"isAllowAdd"#false,"isCurrentLayer"#false,"isVisible"#true,"isDeleted"#true,"isAllowUp"#false,"isAllowDown"#false},'
    a_json += '{"layerId"#1,"name"#"Layer 1","isBackgroundLayer"#false,"isAllowAdd"#false,"isCurrentLayer"#false,"isVisible"#true,"isDeleted"#true,"isAllowUp"'
    a_json += '#false,"isAllowDown"#false},{"layerId"#0,"name"#"Main Layer","isBackgroundLayer"#false,"isAllowAdd"#false,"isCurrentLayer"#true,"isVisible"#true,'
    a_json += '"isDeleted"#false,"isAllowUp"#false,"isAllowDown"#false},{"layerId"#-1,"name"#"Background '
    a_json += 'Layer","isBackgroundLayer"#true,"isAllowAdd"#true,"isCurrentLayer"#false,"isVisible"#true,"isDeleted"#false,"isAllowUp"#false,"isAllowDown"#false}]>'
    a_json += f'<LAYERSEQ:MAINLAYER,BGLAYER><MAINLAYER:{main_layer_address}><LAYER1:0><LAYER2:0><LAYER3:0><BGLAYER:{background_layer_address}><TOTALPATH:{totalpath_address}>'
    a_json += f'<THUMBNAILTYPE:0><RECOGNSTATUS:{rec_status}><RECOGNTEXT:{rec_address}><RECOGNFILE:0><PAGEID:{page_id}><RECOGNTYPE:{rec_page_type}><RECOGNFILESTATUS:{recf_status}><RECOGNLANGUAGE:none>'
    a_json += '<EXTERNALLINKINFO:0><IDTABLE:0><ORIENTATION:1000>'
    a_json_ = a_json.encode('utf-8')
    json_s = int_to_little_endian_bytes(len(a_json_))
    result = json_s + a_json_
    return result


def sn_bytes_page_recap(
    page_recap_location, page_address_dict, placeholder_image_bloc_address,
        apply_equipment=NOTEBOOK_DEVICE, templates_dict={}, titles_recap='', page_keyword_dict={}):
    """ Generates json for the page recap dict, including prefix size, tail and validator"""
    try:
        a_json = ''
        for a_page_key, a_page_value in page_address_dict.items():
            a_json += f'<PAGE{a_page_key}:{a_page_value["pg_address"]}>'

        if titles_recap != '':
            a_json += titles_recap

        for a_page_key, a_page_value in page_keyword_dict.items():
            a_json += f'<TITLE_{a_page_key}:{a_page_value["pg_address"]}>'

        a_json += '<COVER_0:0><DIRTY:0><FILE_FEATURE:24>'
        added_styles_list = []
        for a_page_key, a_page_value in page_address_dict.items():

            if a_page_key in templates_dict:
                a_page_templates_dict = templates_dict[a_page_key]
                pdfstylelist_dict = a_page_templates_dict['pdfstylelist']

                style_name_ = pdfstylelist_dict['name'].split('_')

                style_name = f'{style_name_[0]}_{style_name_[1]}{style_name_[3]}_{style_name_[4]}'
                if style_name not in added_styles_list:
                    added_styles_list.append(style_name)
                    style_rle_address = a_page_templates_dict['address']
                    a_json += f'<STYLE_{style_name}:{style_rle_address}>'

            if apply_equipment == 'N5':
                if 'style_white_a5x2' not in added_styles_list:
                    added_styles_list.append('style_white_a5x2')
                    a_json += f'<STYLE_style_white_a5x2:{placeholder_image_bloc_address}>'
            else:
                if 'style_white' not in added_styles_list:  # MMB 241120
                    added_styles_list.append('style_white')
                    a_json += f'<STYLE_style_white:{placeholder_image_bloc_address}>'

        a_json += 'tail'
        a_json_ = a_json.encode('utf-8')
        json_s = int_to_little_endian_bytes(len(a_json_)-4)
        result = json_s + a_json_ + int_to_little_endian_bytes(page_recap_location)
    except Exception as e:
        print(f'*** sn_bytes_page_recap: {e}')
    return result


def datetime_identifier(file_id=None):
    """ Returns the current datetime as a 20 characters long string"""
    if file_id is None:
        original_uuid = uuid.uuid4()
        file_id_ = str(base64.encodebytes(original_uuid.bytes))
        file_id_ab = ''.join([x for x in file_id_ if x in 'AbcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'])
        file_id = file_id_ab[2:14]
    return str(datetime.now().strftime("%Y%m%d%H%M%S%f") + file_id)


def signed_to_bytes(a_number, num_bytes, byteorder='little'):
    """ Converts a positive number to bytes. If negative, returns a zero bytes"""
    a_signed = a_number < 0
    return a_number.to_bytes(num_bytes, byteorder=byteorder, signed=a_signed)


def decimal_to_custom_ieee754(value, offset=0, num_bytes=4):
    try:
        result = None
        # Adjust the value by the offset
        adjusted_value = value - offset

        # Handle special case for 0
        if adjusted_value == 0:
            return bytes.fromhex('00000000')

        # Determine the sign
        sign = 0 if adjusted_value >= 0 else 1
        adjusted_value = abs(adjusted_value)

        # Calculate the exponent and mantissa
        exponent = 0
        mantissa = adjusted_value

        while mantissa >= 2:
            mantissa /= 2
            exponent += 1

        while mantissa < 1:
            mantissa *= 2
            exponent -= 1

        # Adjust exponent for bias
        exponent += 127

        # Convert mantissa to binary string (23 bits)
        mantissa_str = ''
        mantissa -= 1  # Remove the leading 1
        for _ in range(23):
            mantissa *= 2
            bit = int(mantissa)
            mantissa_str += str(bit)
            mantissa -= bit

        # Construct the binary string
        binary_str = f"{sign:01b}{exponent:08b}{mantissa_str}"

        # Reorder the bytes as per the custom format
        byte4 = binary_str[0:8]
        byte3 = binary_str[8:16]
        byte1 = binary_str[16:24]
        byte2 = binary_str[24:32]

        # Convert to hexadecimal
        result = f"{int(byte1, 2):02X}{int(byte2, 2):02X}{int(byte3, 2):02X}{int(byte4, 2):02X}"
    except Exception as e:
        print(f'*** decimal_to_custom_ieee754: {e}')

    return bytes.fromhex(result)


def shrink_points_relative(points, h, v):
    """
    Shrinks a list of points relative to the first point.

    The first point remains unchanged, and each subsequent point is
    scaled relative to that fixed starting point.

    For each point (x, y), the new coordinates are computed as:
      new_x = start_x + (x - start_x) * (h/100)
      new_y = start_y + (y - start_y) * (v/100)

    Parameters:
      points (list of tuple): List of (x, y) points.
      h (float): Horizontal shrink percentage (e.g., 80 means 80% of the original horizontal distance).
      v (float): Vertical shrink percentage (e.g., 50 means 50% of the original vertical distance).

    Returns:
      list of tuple: The list of transformed (shrunk) points.
    """
    if not points:
        return []

    start_x, start_y = points[0]
    h_factor = h / 100.0
    v_factor = v / 100.0

    new_points = []
    for (x, y) in points:
        new_x = start_x + (x - start_x) * h_factor
        new_y = start_y + (y - start_y) * v_factor
        new_points.append((new_x, new_y))
    return new_points


def text_to_pen_strokes_nf(
        full_dict_ps, titles_dict, images_dict,
        text_to_convert, ascii_ps_list, starting_point, page_nb, temp_dict={},
        xl_json=False, width=None, height=None, delta_x=3, delta_y=12, word_separator=30,
        padding_horizontal=20, padding_vertical=20, not_found_placeholder='#', scratio=1, style_bold_weight=300,
        series=NOTEBOOK_DEVICE, new_color=None, new_weight=None, new_font_size=None,
        font_family=FONT_NAME, screen_ratio=1):
    """ For a given text_to_convert, generates a dictionary of list of pen strokes, per page"""
    try:
        # TODO Pass this as a variable. Here the meaning is:
        # - extend to 1 next word
        # - use ord 170 for style
        # - with thickness 0.5
        # - background color to be used for table of contents = 217
        # Get the ascii_ps_list
        global FONT_RATIOS
        auto_title_dict = TEXT2NOTES_STYLING
        sum_letter_width = 0
        sum_letter_height = 0
        count_letter_width = 0

        markdown_length_dict = {}
        markdown_length_list = []
        if series in ['N5']:
            a_v_offset_ratio = MAX_VERTICAL_PIXELS_N5 / MAX_VERTICAL_PIXELS_N5
        else:
            a_v_offset_ratio = 1

        try:
            ratio_x = ratio_y = 1
            fonts_adj_json_fn = os.path.join(CONFIG, f'{font_family}.json')
            fonts_adjustment = read_json(fonts_adj_json_fn)

            fonts_narrowing_after = fonts_adjustment['narrowing_after']
            if fonts_narrowing_after == {}:
                fonts_narrowing_after = FONTS_NARROWING_AFTER

            fonts_narrowing_before = fonts_adjustment['narrowing_before']
            if fonts_narrowing_before == {}:
                fonts_narrowing_before = FONTS_NARROWING_BEFORE

            if 'font_ratios' in fonts_adjustment:
                FONT_RATIOS = fonts_adjustment['font_ratios']

            if new_font_size:
                if 'font_size' in fonts_adjustment:
                    font_size = fonts_adjustment['font_size']
                    ratio_x = round(FONT_RATIOS[0]*new_font_size/font_size, 2)
                    ratio_y = round(FONT_RATIOS[1]*new_font_size/font_size, 2)

            fonts_vertical_adjustment_ = list_to_tree(fonts_adjustment['vertical'])[ratio_y]

            interval = next(iter(fonts_vertical_adjustment_))
            fonts_vertical_adjustment = interval.data

            if fonts_vertical_adjustment == {}:
                fonts_vertical_adjustment = CHAR_V_OFFSET

        except Exception as e:
            print(f'*** Vertical adjustment: {e}')
            print(f' >  File: {fonts_adj_json_fn}. Using default')
            fonts_vertical_adjustment = CHAR_V_OFFSET
            fonts_narrowing_after = FONTS_NARROWING_AFTER
            fonts_narrowing_before = FONTS_NARROWING_BEFORE

        l_rows = text_to_convert.split('\n')

        markdown_length_list = [(x, len(x.split())-1) for x in l_rows if ((x != '') and (x[:2] == '^h' or x[:2] == '^l'))]

        # Build the dictionary of headings
        for x in markdown_length_list:
            key_word_l = x[0].split()

            key_word = key_word_l[0].lower()
            heading_next_nb = x[1]
            if key_word in markdown_length_dict:
                full_heading_list = markdown_length_dict[key_word]
                full_heading_list.append({"full": x[0][3:], "next": heading_next_nb})
            else:
                markdown_length_dict[key_word] = [{"full": x[0][3:], "next": heading_next_nb}]

        # text_to_convert = text_to_convert.replace('\n', ' ^p ').replace('  ', ' ')
        # text_to_convert = text_to_convert.replace('\n', '^p')
        text_to_convert = re.sub(r'\n', '^p', text_to_convert)

        # text_to_convert_list = text_to_convert.split()

        text_to_convert_list = [text_to_convert]

        nb_words = len(text_to_convert_list)

        dict_ps = {}

        list_ps = []
        list_xl_points = []
        list_titles_image = []
        starting_pos_x = starting_point[0] + new_font_size
        starting_pos_y = starting_point[1]

        max_iterations_loop = MAX_PEN_TO_STROKE_LOOP
        style_indent = 0
        pos_x = starting_pos_x
        pos_y = starting_pos_y

        index_word = 0
        index_letter = 0
        current_word_ps_count = 0
        max_row_height = 0
        nb_loops = 0
        page_number = 0
        title_ps = []
        title_ps_c = {}
        list_all_x = []
        list_all_y = []
        list_cw_reset = -1
        style_char = ''
        style_prefix = ''
        style_creturn = 0
        pdf_page_nb = 0
        style_weight = None
        style_post_word_nb = 1
        if '^b1' in auto_title_dict:
            style_bold_weight_settings = auto_title_dict['^b1']
            if 'weight' in style_bold_weight_settings:
                style_bold_weight = NEEDLE_POINT_SIZES[str(style_bold_weight_settings['weight'])]
                # style_weight = style_bold_weight
        style_type = ''
        style_bold_on = False

        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)

        max_horizontal_length_ps = max_horizontal_pixels/scratio - padding_horizontal - round(
            adb_screen_max_x*(1-scratio)*(1-MARGIN_LEFT_SIZE))
        style_char_ps = None  # MMB 241130
        while (index_word < nb_words) and (nb_loops < max_iterations_loop):

            current_word = text_to_convert_list[index_word]

            if '$Page' in current_word:
                pdf_page_nb = text_to_convert_list[index_word+1]

            current_word_ps_count = 0
            current_word_ps = []
            # style_char_ps = None   MMB 241130

            # The If block below breaks long words in pieces and restarts the loop
            # if (len(current_word)*20 + pos_x > max_horizontal_length_ps):
            if (pos_x > max_horizontal_length_ps):

                pos_x = starting_pos_x

                # Determining where to split
                split_position = round((max_horizontal_length_ps-pos_x)/20/2)

                current_word_1 = current_word[:split_position]
                current_word_2 = current_word[split_position:]

                # Updating the list of words by inserting a 2nd piece, updating the number of words
                if current_word_2 != '':
                    text_to_convert_list[index_word] = current_word_1
                    text_to_convert_list.insert(index_word+1, current_word_2)
                    nb_words += 1
                nb_loops += 1

                # Increment the vertical position
                pos_y += max_row_height + delta_y

                # Check if new page is needed TODO: set this as a function because code repeated elsewhere
                if pos_y > max_vertical_pixels/scratio - padding_vertical:

                    if len(list_ps) > 0:

                        if len(current_word_ps) > 0:
                            _temp_list = []
                            for sub_list in current_word_ps:
                                _temp_list.extend(sub_list)
                            list_ps.extend(_temp_list)

                        current_word_ps_count = len(current_word_ps)
                        if current_word_ps_count > 0:
                            list_ps = list_ps[:-1*current_word_ps_count]
                        # index_word -= 1
                        index_letter = 0
                        current_word_ps_count = 0
                        current_word_ps = []
                        current_letter_ps = []
                        current_letter_ps_ = []
                        style_char_ps_ = []
                        pdf_page_nb = 0

                        dict_ps[f'{page_nb}'] = list(list_ps)

                        list_ps = []

                        page_number += 1
                    pos_x = starting_pos_x
                    pos_y = starting_pos_y
                    list_ps = []
                    list_titles_image = []
                continue

            # Initialize current word ps list
            current_word_ps = []
            current_xl_points = []
            style_case = None

            index_letter = 0

            if current_word == '':
                pos_x += word_separator
                index_word += 1
                continue
            elif current_word == '^p':
                pos_x = starting_pos_x
                pos_y += max_row_height + delta_y
                if pos_y > max_vertical_pixels/scratio - padding_vertical:
                    if len(list_ps) > 0:

                        if len(current_word_ps) > 0:
                            _temp_list = []
                            for sub_list in current_word_ps:
                                _temp_list.extend(sub_list)
                            list_ps.extend(_temp_list)

                        current_word_ps_count = len(current_word_ps)
                        if current_word_ps_count > 0:
                            list_ps = list_ps[:-1*current_word_ps_count]
                        # index_word -= 1
                        index_letter = 0
                        current_word_ps_count = 0
                        current_word_ps = []
                        current_letter_ps = []
                        current_letter_ps_ = []
                        style_char_ps_ = []

                        dict_ps[f'{page_nb}'] = list(list_ps)

                        list_ps = []

                        page_number += 1
                    pos_x = starting_pos_x
                    pos_y = starting_pos_y
                    list_ps = []
                    list_titles_image = []
                index_word += 1
                continue

            cw_lower = current_word.lower()

            # Indicates whether or not the styling is ON
            get_xt_point = index_word <= list_cw_reset

            # This if condition below is entered when the styling is turned OFF
            if not get_xt_point:
                # style_bold_on = False
                list_all_x = []
                list_all_y = []
                list_cw_reset = -1
                title_ps = []
                title_ps_c = {}
                style_weight = None
                style_case_upper = style_case_lower = False

                pos_y += style_creturn*20
                if pos_y > max_vertical_pixels/scratio - padding_vertical:
                    if len(list_ps) > 0:

                        if len(current_word_ps) > 0:
                            _temp_list = []
                            for sub_list in current_word_ps:
                                _temp_list.extend(sub_list)
                            list_ps.extend(_temp_list)

                        current_word_ps_count = len(current_word_ps)
                        if current_word_ps_count > 0:
                            list_ps = list_ps[:-1*current_word_ps_count]
                        # index_word -= 1
                        index_letter = 0
                        current_word_ps_count = 0
                        current_word_ps = []
                        current_letter_ps = []
                        current_letter_ps = []
                        current_letter_ps_ = []
                        style_char_ps_ = []

                        dict_ps[f'{page_nb}'] = list(list_ps)

                        list_ps = []

                        page_number += 1
                    pos_x = starting_pos_x
                    pos_y = starting_pos_y
                    list_ps = []
                    list_titles_image = []
                style_creturn = 0

                style_indent = 0

                # If we find cw_lower in the list of styles, we load the parameters
                # print(f'------cw_lower: {cw_lower}')
                if cw_lower in auto_title_dict:
                    style_type = 'none'
                    style_settings = auto_title_dict[cw_lower]
                    # style_ord = style_settings["ord"]     # Read the value of the unicode in the dictionary
                    if 'ord' in style_settings:
                        style_ord = style_settings["ord"]   # Read the value of the unicode in the dictionary
                        style_char = chr(style_ord)         # The "char" for the style
                        style_char_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                            ascii_ps_list, temp_dict=temp_dict,
                            ratio_x=ratio_x, ratio_y=ratio_y, bypassed_unicode=style_ord, device=series,
                            font_family=font_family)
                    else:
                        style_ord = None
                        style_char = ''
                        style_char_ps = None

                    if 'toc_bg_color' in style_settings:
                        style_bg_color_ = style_settings["toc_bg_color"]
                        style_bg_color = (style_bg_color_, style_bg_color_, style_bg_color_)

                    if 'weight' in style_settings:
                        style_weight = NEEDLE_POINT_SIZES[str(style_settings['weight'])]  # See NEEDLE_POINT_SIZES
                    else:
                        style_weight = None
                    style_char_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                        ascii_ps_list, temp_dict=temp_dict,
                        ratio_x=ratio_x, ratio_y=ratio_y, bypassed_unicode=style_ord, device=series, font_family=font_family)

                    if 'next_word_count' in style_settings:
                        style_post_word_nb = auto_title_dict[cw_lower]['next_word_count']  # We extend the style to x following words
                    list_cw_reset = style_post_word_nb + index_word
                    get_xt_point = index_word <= list_cw_reset
                    if current_word[0] == '$':
                        current_word = current_word[1:]

                # ------- Test if markdown heading -----------------------------------------

                elif cw_lower in markdown_length_dict:
                    a_list = markdown_length_dict[cw_lower]
                    if a_list != []:
                        first_element = a_list.pop(0)
                        # first_element_full = first_element['full']
                        style_post_word_nb = first_element['next']

                        list_cw_reset = index_word + style_post_word_nb

                        get_xt_point = index_word <= list_cw_reset

                        mkd_mark = cw_lower[:3]
                        # print(f'-----mkd_mark: {mkd_mark}')

                        # If we find mkd_mark in the list of styles, we load the parameters
                        if mkd_mark in auto_title_dict:

                            if mkd_mark[1] == 'l':
                                style_type = 'bullet-point'
                            elif mkd_mark[1] == 'h':
                                style_type = 'heading'
                            elif mkd_mark[1:] == 'pa':
                                style_type = 'heading'
                            else:
                                style_type = 'none'

                            style_settings = auto_title_dict[mkd_mark]

                            if 'ord' in style_settings:
                                style_ord = style_settings["ord"]   # Read the value of the unicode in the dictionary
                                style_char = chr(style_ord)         # The "char" for the style
                                style_char_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                                    ascii_ps_list, temp_dict=temp_dict,
                                    ratio_x=ratio_x, ratio_y=ratio_y, bypassed_unicode=style_ord, device=series,
                                    font_family=font_family)
                            else:
                                style_ord = None
                                style_char = ''
                                style_char_ps = None

                            if 'toc_bg_color' in style_settings:
                                style_bg_color_ = style_settings["toc_bg_color"]
                                style_bg_color = (style_bg_color_, style_bg_color_, style_bg_color_)
                            else:
                                style_bg_color = (254, 254, 254)

                            if 'weight' in style_settings:
                                style_weight = NEEDLE_POINT_SIZES[str(style_settings['weight'])]  # See NEEDLE_POINT_SIZES
                            else:
                                style_weight = None

                            if 'prefix' in style_settings:
                                style_prefix = style_settings['prefix']
                            else:
                                style_prefix = ''

                            # Setting up case if relevant
                            if 'case' in style_settings:
                                style_case = style_settings['case']
                                style_case_upper = style_case.lower() in 'uppercase'
                                style_case_lower = style_case.lower() in 'lowercase'

                            # Setting up carriage return if relevant
                            if 'creturn' in style_settings:
                                style_creturn = style_settings['creturn']
                                pos_y += style_creturn*20
                                if pos_y > max_vertical_pixels/scratio - padding_vertical:
                                    if len(list_ps) > 0:

                                        if len(current_word_ps) > 0:
                                            _temp_list = []
                                            for sub_list in current_word_ps:
                                                _temp_list.extend(sub_list)
                                            list_ps.extend(_temp_list)

                                        current_word_ps_count = len(current_word_ps)
                                        if current_word_ps_count > 0:
                                            list_ps = list_ps[:-1*current_word_ps_count]
                                        # index_word -= 1
                                        index_letter = 0
                                        current_word_ps_count = 0
                                        current_word_ps = []
                                        current_letter_ps = []
                                        current_letter_ps_ = []
                                        style_char_ps_ = []

                                        dict_ps[f'{page_nb}'] = list(list_ps)

                                        list_ps = []

                                        page_number += 1
                                    pos_x = starting_pos_x
                                    pos_y = starting_pos_y
                                    list_ps = []
                                    list_titles_image = []
                            # Setting up indent if relevant
                            if 'indent' in style_settings:
                                style_indent = -1 * style_settings['indent'] * 50
                                pos_x += style_indent

            # Bullet points are in bold only if specifically marked so
            if current_word[:2] == '^l':
                if len(current_word) > 5:
                    style_bold_on = current_word[3:6] == '^b1'

            replaced_hl = False
            # print(f'------current_word:{current_word}')
            if current_word[:3] in ['^h1', '^h2', '^h3', '^h4', '^h5', 'h6', '^l1', '^l2', '^Pa']:
                style_bold_on = False  # MMB
                replaced_hl = True
                current_word = current_word[3:]

                # print(f'-----current_word:{current_word}')
            if current_word[:3] == '^b1':
                style_bold_on = not style_bold_on  # True
                current_word = current_word[3:]

            elif index_word > 0:
                previous_w = text_to_convert_list[index_word-1]
                if previous_w[-3:] == '^b1':
                    style_bold_on = False  # This is correct only if the bold style was set for the previous word

            if replaced_hl:
                current_word = style_prefix + current_word

            current_word = current_word.replace('^b1', '')
            if '$Page' in current_word:
                current_word = current_word.replace('$', '')
            index_letter = 0
            while index_letter < len(current_word):

                xt_rect = []
                current_letter = current_word[index_letter]
                if current_letter in fonts_narrowing_after:
                    horizontal_offset = fonts_narrowing_after[current_letter]
                else:
                    horizontal_offset = 0

                if current_letter in fonts_narrowing_before:
                    horizontal_offset_before = fonts_narrowing_before[current_letter]
                else:
                    horizontal_offset_before = 0

                if current_letter == " ":
                    pos_x += delta_x + word_separator
                    index_letter += 1
                    continue

                if style_case_upper:
                    current_letter = current_letter.upper()
                elif style_case_lower:
                    current_letter = current_letter.lower()

                current_letter_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                    ascii_ps_list, searched_char=current_letter, temp_dict=temp_dict,
                    ratio_x=ratio_x, ratio_y=ratio_y, device=series, font_family=font_family)

                # If the character is not found, replace by the placeholder_char
                if current_letter_ps == []:
                    ascii_equivalent = unidecode(current_letter)
                    current_letter_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                        ascii_ps_list, searched_char=ascii_equivalent[0], temp_dict=temp_dict,
                        ratio_x=ratio_x, ratio_y=ratio_y, device=series, font_family=font_family)
                    if current_letter_ps == []:
                        print(f'**- char_from_ascii_tps: "{current_letter}" not found. Replacing with "{not_found_placeholder}"')
                        current_letter = not_found_placeholder
                        current_letter_ps, stroke_width, stroke_height, temp_dict = char_from_ascii_tps(
                            ascii_ps_list, searched_char=current_letter, temp_dict=temp_dict,
                            ratio_x=ratio_x, ratio_y=ratio_y, device=series, font_family=font_family)
                        # print(current_letter_ps)
                        # exit(1)

                current_word_ps_count += len(current_letter_ps)

                # Retrieve vertical offset adjustments, if needed
                if current_letter in fonts_vertical_adjustment:
                    a_v_offset = fonts_vertical_adjustment[current_letter]*ratio_y
                else:
                    a_v_offset = 0

                # Reposition the current_letter_ps based on its position
                if get_xt_point:
                    if style_type == 'heading':
                        new_weight = style_weight
                    else:
                        if style_bold_on:
                            new_weight = style_bold_weight

                else:
                    if style_bold_on:
                        new_weight = style_bold_weight

                # print(f'--{current_letter}-w={stroke_width}-horizontal_offset_before:{horizontal_offset_before}')
                pos_x -= stroke_width*horizontal_offset_before

                current_letter_ps_, letter_width, letter_height, xt_rect = strokes_at_point_nf(
                    current_letter_ps, pos_x, pos_y + a_v_offset*a_v_offset_ratio+new_font_size*screen_ratio,  get_xt_point=get_xt_point, new_weight=new_weight,
                    series=series, new_color=new_color, width=width, height=height)

                style_char_ps_ = []
                if get_xt_point:
                    list_all_x.append(xt_rect[0])
                    list_all_y.append(xt_rect[1])
                    list_all_x.append(xt_rect[2])
                    list_all_y.append(xt_rect[3])
                    # reposition the style char ps
                    # Retrieve vertical offset adjustments, if needed
                    if style_char in fonts_vertical_adjustment:
                        a_v_offset = fonts_vertical_adjustment[style_char]*ratio_y
                    else:
                        a_v_offset = 0
                    if style_char_ps:
                        style_char_ps_, _, _, _ = strokes_at_point_nf(
                            style_char_ps, pos_x, pos_y + a_v_offset*a_v_offset_ratio+new_font_size*screen_ratio, new_color=new_color,
                            get_xt_point=False, series=series)

                current_word_ps.append(style_char_ps_)
                current_word_ps.append(current_letter_ps_)

                # Compute the current max_row_height
                max_row_height = max(max_row_height, letter_height)

                # Compute the current horizontal position
                if style_weight:
                    delta_multiplier = round(3*(style_weight-400)/400)
                else:
                    delta_multiplier = 0
                pos_x += letter_width*(1-horizontal_offset) + delta_x*(1+delta_multiplier)
                sum_letter_width += letter_width
                sum_letter_height += letter_height
                count_letter_width += 1

                carriage_return = False
                nb_cr = 0
                while (index_letter < len(current_word) - 2) and (current_word[index_letter+1:index_letter+3] == '^p'):
                    carriage_return = True
                    index_letter += 2
                    nb_cr += 1

                # If the horizontal position is getting offscreen, break the line
                if carriage_return or (pos_x > max_horizontal_length_ps):
                    # if carriage_return or (pos_x > max_horizontal_length_ps):
                    # TODO: the formula under "round()" is incorrect - MAX_HORIZONTAL_PIXELS*MARGIN_LEFT_SIZE*(1-scratio)
                    # Go to next vertical row

                    pos_y += max_row_height + delta_y
                    if nb_cr > 1:
                        for i in range(0, nb_cr-1):
                            pos_y += max_row_height + delta_y

                    if pos_y > max_vertical_pixels/scratio - padding_vertical:

                        if len(list_ps) > 0:

                            if len(current_word_ps) > 0:
                                _temp_list = []
                                for sub_list in current_word_ps:
                                    _temp_list.extend(sub_list)
                                list_ps.extend(_temp_list)

                            current_word_ps_count = len(current_word_ps)

                            if current_word_ps_count > 0:
                                list_ps = list_ps[:-1*current_word_ps_count]
                            # index_word -= 1
                            index_letter = 0
                            current_word_ps_count = 0
                            current_word_ps = []
                            current_letter_ps = []
                            current_letter_ps_ = []
                            style_char_ps_ = []

                            dict_ps[f'{page_nb}'] = list(list_ps)

                            list_ps = []

                            page_number += 1

                        else:
                            print(f'----list_ps was empty for word: {current_word} ')
                            print(f'------len(current_word_ps): {len(current_word_ps)}')
                            print(f'-------page_number: {page_number}')

                        pos_x = starting_pos_x
                        pos_y = starting_pos_y
                        list_ps = []
                        list_titles_image = []

                    # Reposition to starting horizontal position
                    pos_x = starting_pos_x

                    current_xl_points = []

                    if current_letter != style_char:
                        if len(current_letter_ps_) > 0:
                            title_color = str(current_letter_ps_[0]['color'])
                        l_title_vl = [x['vector_points'] for x in current_letter_ps_]

                        if title_color in title_ps_c:
                            a_color_list = title_ps_c[title_color]
                            a_color_list.extend(l_title_vl)
                        else:
                            title_ps_c[title_color] = l_title_vl
                        title_ps.extend(l_title_vl)
                    index_letter += 1
                    max_row_height = 0
                    continue

                # There is still room horizontally
                else:

                    if current_letter != style_char:
                        if len(current_letter_ps_) > 0:
                            title_color = str(current_letter_ps_[0]['color'])
                        l_title_vl = [x['vector_points'] for x in current_letter_ps_]

                        if title_color in title_ps_c:
                            a_color_list = title_ps_c[title_color]
                            a_color_list.extend(l_title_vl)
                        else:
                            title_ps_c[title_color] = l_title_vl
                        title_ps.extend(l_title_vl)

                    # Add all the individual points of the character strokes for XL
                    if xl_json:
                        for a_ps in current_letter_ps_:
                            for x in a_ps['vector_points']:
                                current_xl_points.append((x[0], x[1]))

                index_letter += 1

            if len(current_word_ps) > 0:
                _temp_list = []
                for sub_list in current_word_ps:
                    _temp_list.extend(sub_list)
                list_ps.extend(_temp_list)
                current_word_ps = []
                current_letter_ps = []
                current_letter_ps_ = []
                style_char_ps_ = []

                if xl_json:
                    list_xl_points.extend(current_xl_points)

                index_word += 1
                nb_loops = 0
                pos_x += delta_x + word_separator
            else:
                nb_loops += 1
                index_word += 1

            if list_all_x != []:
                if len(current_letter_ps_) > 0:
                    title_color = str(current_letter_ps_[0]['color'])
                    l_title_vl = [x['vector_points'] for x in current_letter_ps_]
                    title_ps.extend(l_title_vl)
                    if title_color in title_ps_c:
                        a_color_list = title_ps_c[title_color]
                        a_color_list.extend(l_title_vl)
                    else:
                        title_ps_c[title_color] = l_title_vl

                min_xt_x = min(list_all_x)
                min_xt_y = min(list_all_y)
                max_xt_x = max(list_all_x)
                max_xt_y = max(list_all_y)

                off_minx_v = 0
                off_miny_v = -10
                off_maxx_v = 10
                off_maxy_v = 10

                min_xt_x_v = round((adb_screen_max_x+16-max_xt_x)*(max_horizontal_pixels/adb_screen_max_x)+off_minx_v)
                min_xt_y_v = round((min_xt_y+32)*(max_vertical_pixels/adb_screen_max_y)+off_miny_v)

                hl_list = [
                    min_xt_x_v,
                    min_xt_y_v,
                    round((adb_screen_max_x+16-min_xt_x)*(max_horizontal_pixels/adb_screen_max_x)+off_maxx_v)-min_xt_x_v+25,
                    round((max_xt_y+32)*(max_vertical_pixels/adb_screen_max_y)+off_maxy_v)-min_xt_y_v]

                page_str = str(page_nb)

                if index_word == list_cw_reset + 1:

                    # We only save images for toc for heading, not for bullet-points
                    if style_type == 'heading':
                        if page_str in titles_dict:

                            page_titles = titles_dict[page_str]

                            a_title_dict = {"rect": hl_list, "ps": title_ps_c, "pdf_page": pdf_page_nb}
                            page_titles.append(a_title_dict)
                            titles_dict[page_str] = page_titles
                        else:
                            titles_dict[page_str] = [
                                {
                                    "rect": hl_list,
                                    "ps": title_ps_c,
                                    "pdf_page": pdf_page_nb}
                                    ]
                    style_type = ''

        if nb_loops >= max_iterations_loop:
            print()
            print('**- text_to_pen_strokes_nf: WARNING: A likely infinite while loop was detected and exited')
            print(text_to_convert_list)

        page_nb_str = str(page_nb)

        if list_ps != []:
            dict_ps[page_nb_str] = list(list_ps)
            list_ps = []

        for a_page, a_list_ps in dict_ps.items():
            if page_nb_str in full_dict_ps:
                current_full_dict_ps_page = full_dict_ps[page_nb_str]
                current_full_dict_ps_page['strokes_nb'] = current_full_dict_ps_page['strokes_nb'] + len(a_list_ps)
                current_full_dict_ps_page['strokes'].extend(a_list_ps)
                current_full_dict_ps_page['last_position'] = [
                    pos_x + letter_width + delta_x + word_separator, pos_y]
                full_dict_ps[page_nb_str] = current_full_dict_ps_page
            else:
                full_dict_ps[page_nb_str] = {
                    "strokes_nb": len(a_list_ps),
                    "strokes": a_list_ps,
                    "last_position": [pos_x + letter_width + delta_x + word_separator, pos_y]}

        list_xl_points = list(set(list_xl_points))
        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)
        for a_page, a_pagetitle_ps_list in titles_dict.items():

            list_titles_image = []

            for a_pagetitle in a_pagetitle_ps_list:
                a_pagetitle_rect = a_pagetitle['rect']
                a_pagetitle_ps = a_pagetitle['ps']
                a_pagetitle_pdf = a_pagetitle['pdf_page']

                new_dict = {}
                for tpk_key, tps_value in a_pagetitle_ps.items():
                    converted_list = []

                    for an_el in tps_value:
                        an_el_converted = [
                            topright_to_topleft(
                                x, max_horizontal_pixels, max_vertical_pixels,
                                adb_screen_max_x, adb_screen_max_y) for x in an_el]
                        converted_list.append(an_el_converted)

                    new_dict[tpk_key] = list(converted_list)

                an_image = draw_bitmap_from_vectors(new_dict, (max_horizontal_pixels, max_vertical_pixels), image_crop=a_pagetitle_rect, background_color=style_bg_color)

                rleimage, encoded = rle_encode_img(an_image)

                if encoded:
                    rleimage_size = len(rleimage)
                    placeholder_image_size = int_to_little_endian_bytes(rleimage_size)
                    result = placeholder_image_size + rleimage

                    list_titles_image.append([a_pagetitle_rect, result.hex(), a_pagetitle_pdf])

            images_dict[a_page] = list(list_titles_image)

        if xl_json:
            new_list_xl_points = []
            for a_point in list_xl_points:
                new_list_xl_points.extend([a_point[0], a_point[1]])
            return full_dict_ps, new_list_xl_points, images_dict
        return full_dict_ps, None,  images_dict
    except Exception as e:
        print()
        print(f'*** text_to_pen_strokes_nf: {e}')
        if xl_json:
            return None, None, None
        return None, None, None


def psdict_to_note(ps_dict, titles_dict, output_fn, starting_point=None, scratio=1, series=NOTEBOOK_DEVICE, sn_pdf_fn=''):
    """ Create a json with pen_strokes corresponding to text

    Parameters:
        - Starting_point: Pixels coordinate of the first print on the notebook
        - text_to_convert: Text to convert
        - output_fn: Name of output file in a .Note format

    Returns: a dictionary with list of strokes for each page
        """
    try:

        if starting_point is None:
            if series in ['N5']:
                starting_point = [10, 250]
            else:
                starting_point = [7, 183]

        if DEBUG_MODE:
            save_json(output_fn+'_titles_t2notes.json', titles_dict)

        # ---- Binary .Note file building ---------

        file_signature = sn_bytes_header()  # Generate the file signature
        file_id, first_json, _ = sn_bytes_1_header_json(apply_equipment=series)  # Generate fileid and 1st json

        # placeholder_image_bloc = sn_bytes_2_basic_still_image(placeholder_bitmap_fn)
        placeholder_image_bloc = blank_encoded_image(series)

        # Load the dictionary of binary representations of totalpath, per page
        totalpath_bytes = pen_strokes_dict_to_bytes(ps_dict, series=series)

        image_layer_dict = generate_ephemeral_images(ps_dict, series=series)

        first_bytes_before_image = file_signature + first_json

        current_bytes = first_bytes_before_image

        placeholder_image_bloc_address = len(current_bytes)
        current_bytes += placeholder_image_bloc

        main_dict = {}
        for ild_p, ild_v in image_layer_dict.items():
            ild_bitmap_address = len(current_bytes)
            current_bytes += ild_v
            ild_main = sn_bytes_layer_json(ild_bitmap_address, name='MAINLAYER')
            ild_main_address = len(current_bytes)
            current_bytes += ild_main
            main_dict[ild_p] = ild_main_address
        background_layer_address = len(current_bytes)

        background_layer = sn_bytes_layer_json(placeholder_image_bloc_address, name='BGLAYER')
        current_bytes += background_layer

        current_bytes, titles_recap = build_titles_image_n_details(titles_dict, current_bytes)

        # print(f'----titles_dict:{titles_dict}')

        # Build the page links, if applicable
        title_links_recap = ''
        if sn_pdf_fn != '':

            for sn_page_nb, list_title_links in titles_dict.items():
                for a_title_link in list_title_links:
                    link_rect, _, pdf_page_nb = a_title_link
                    link_name, link_json_address, current_bytes = generate_link_json(
                        sn_page_nb, sn_pdf_fn, pdf_page_nb, link_rect, current_bytes)
                    title_links_recap += f'<{link_name}:{link_json_address}>'

        titles_recap += title_links_recap

        # Parsing the pages and add totalpath bytes for each page
        totalpath_address_dict = {}
        for a_page_nb, page_totalpath_bytes in totalpath_bytes.items():
            totalpath_size = len(page_totalpath_bytes)
            totalpath_size_bytes = int_to_little_endian_bytes(totalpath_size)
            totalpath_address_dict[a_page_nb] = {"tp_address": len(current_bytes)}
            current_bytes += totalpath_size_bytes + page_totalpath_bytes

        # Parsing the pages and add page information for each page
        for a_page_nb in totalpath_address_dict:
            a_page_ref_dict = totalpath_address_dict[a_page_nb]
            a_page_total_path = a_page_ref_dict['tp_address']
            a_page_ref_dict['pg_address'] = len(current_bytes)

            main_layer_address = main_dict[a_page_nb]

            page_bytes = sn_bytes_page(main_layer_address, background_layer_address, a_page_total_path)
            current_bytes += page_bytes

        page_recap_location = len(current_bytes)

        recap_pages = sn_bytes_page_recap(
            page_recap_location, totalpath_address_dict, placeholder_image_bloc_address,
            apply_equipment=series, titles_recap=titles_recap)

        current_bytes += recap_pages

        with open(output_fn, 'wb') as file:
            file.write(current_bytes)

    except Exception as e:
        print()
        print(f'*** psdict_to_note: {e}')


def strokes_at_point_nf(
        strokes_list, x, y, row_height=90, scratio=1, get_xt_point=False, new_weight=None,
        new_color=None, series=NOTEBOOK_DEVICE, width=None, height=None):
    """ Returns a modified strokes list for coordinates x,y (ref MAX_HORIZONTAL_PIXELS/MAX_VERTICAL_PIXELS)
        The function first checks the relative delta_x and delta_y for all the
        strokes in the list. It then applies the x/y coordinates to that point
        and shifts the other strokes by their delta.
        We do not check if the shifting movs the strokes out of the screen boundaries

        strokes_list is a list of dictionaries"""
    try:

        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)

        min_x = min([x['min_c_x'] for x in strokes_list])
        max_x = max([x['max_c_x'] for x in strokes_list])
        min_y = min([x['min_c_y'] for x in strokes_list])
        max_y = max([x['max_c_y'] for x in strokes_list])

        stroke_width = max_x - min_x
        stroke_height = max_y - min_y

        # The base reference for positioning letters is the bottom left corner (x-min, y_max)
        # This simplifies baseline  positionning for exceptions such as f, g, p, q, y

        delta_x = x - min_x
        delta_y = y - max_y

        delta_x_mu = -1 * round(delta_x * adb_screen_max_x / max_horizontal_pixels)

        delta_y_mu = round(delta_y * adb_screen_max_y / max_vertical_pixels)

        scratio_x = round(adb_screen_max_x*(1-scratio)*(1-MARGIN_LEFT_SIZE))

        scratio_pix = round(max_horizontal_pixels*(1-scratio)*(1-MARGIN_LEFT_SIZE))

        # Create a deep copy of the list
        mod_strokes_list = copy.deepcopy(strokes_list)

        # Parse the list

        for ps_dict in mod_strokes_list:

            new_min_x = round(scratio*(ps_dict['min_c_x'] + delta_x)) + scratio_pix
            new_max_x = round(scratio*(ps_dict['max_c_x'] + delta_x)) + scratio_pix
            new_min_y = round(scratio*(ps_dict['min_c_y'] + delta_y))
            new_max_y = round(scratio*(ps_dict['max_c_y'] + delta_y))

            if new_weight:
                ps_dict['weight'] = new_weight

            if new_color:
                ps_dict['color'] = new_color

            ps_dict['min_c_x'] = max(min(new_min_x, max_horizontal_pixels), 0)
            ps_dict['min_c_y'] = max(min(new_min_y, max_vertical_pixels), 0)
            ps_dict['max_c_x'] = max(min(new_max_x, max_horizontal_pixels), 0)
            ps_dict['max_c_y'] = max(min(new_max_y, max_vertical_pixels), 0)
            ps_dict['avg_c_x'] = max(min(round(scratio*(ps_dict['avg_c_x'] + delta_x)) + scratio_x, max_horizontal_pixels), 0)
            ps_dict['avg_c_y'] = max(min(round(scratio*(ps_dict['avg_c_y'] + delta_y)), max_vertical_pixels), 0)  # TODO: Based on 1 example. this was unchanged. WHY? confirm
            xt_vector = [
                [
                    max(min(round(scratio*(x[0] + delta_y_mu)), adb_screen_max_y), 0),
                    max(min(round(scratio*(x[1] + delta_x_mu) + scratio_x), adb_screen_max_x), 0)] for x in ps_dict['vector_points']]
            ps_dict['vector_points'] = xt_vector

            if get_xt_point:
                xt_x_list = [x[1] for x in xt_vector]
                xt_y_list = [x[0] for x in xt_vector]
                xt_rect = [min(xt_x_list), min(xt_y_list), max(xt_x_list), max(xt_y_list)]
            else:
                xt_rect = []

            mod_contours = {}
            for contour_nb, contour_value in ps_dict['contours'].items():
                new_list = [
                    [
                        max(min(round(scratio*(x[0] + delta_x) + scratio_pix), max_horizontal_pixels), 0),
                        max(min(round(scratio*(x[1] + delta_y)), max_vertical_pixels), 0)] for x in contour_value]
                mod_contours[contour_nb] = new_list
            ps_dict['contours'] = mod_contours

        return mod_strokes_list, stroke_width, stroke_height, xt_rect
    except Exception as e:
        print()
        print(f'*** strokes_at_point: {e}')
        return None, 0, 0, []


def char_from_ascii_tps(
    ascii_ps_list, temp_dict={}, searched_char='a', bypassed_unicode=None, device=NOTEBOOK_DEVICE,
        index_0=28, y0=0, elsize=10, ratio_x=FONT_RATIOS[0], ratio_y=FONT_RATIOS[1],
        max_horizontal_pixels=MAX_HORIZONTAL_PIXELS_N5, max_vertical_pixels=MAX_VERTICAL_PIXELS_N5,
        adb_screen_max_x=ADB_SCREEN_MAX_X_N5, adb_screen_max_y=ADB_SCREEN_MAX_Y_N5,
        threshold=0.35, font_family=FONT_NAME):
    """ From a list of ascii ps, returns pen strokes corresponding to a given ascii code
        The order the characters are placed on the table, and their loction within the table is very important.
        The characters don't have to be centered in a cell. They do need to be inside the cell, though
        An intersection threshold of 35% is set to detect all strokes in a given cell
        Parameters:
            - ascii_ps_list:    a list of pen stroke dictionaries
            - bypassed_unicode: Integer. If not None, just use the number to locate the cell in the ascii table
            - searched_char:    the character for which we want to retrieve pen strokes
            - index_0:          the ASCII index of the first character in the table
            - x0, y0:           pixel coordinates of the first element in the table
            - side:             size in pixels of one side of a square cell of the table
            - elsize:           number of cells in a row of the table
            - ratio_x, ratio_y  shrink(expand) ratio the letter's pen strokes relative to bottom left corner
            - temp_dict         temporary dictionary storing pen strokes for a letter with a given ratio that has
                                already been identified in this session. It saves the economy of finding the intersections.
        """
    try:
        # Storage key of a letter with particular ratio in the temporary dictionary
        a_key = f'{font_family}-{searched_char}-{ratio_x}-{ratio_y}'
        if temp_dict != {}:
            if a_key in temp_dict:
                # Returning existing key in the temporary dictionary
                select_strokes, stroke_width, stroke_height = temp_dict[a_key]
                return select_strokes, stroke_width, stroke_height, temp_dict

        if device in ['N5']:
            x0 = ASCII_X02
            side = ASCII_SIDE2
        else:
            x0 = ASCII_X0
            side = ASCII_SIDE

        if bypassed_unicode is None:
            if searched_char in CUSTOM_MAP:
                ascii_index = CUSTOM_MAP[searched_char]
            else:
                ascii_index = ord(searched_char)
        else:
            ascii_index = bypassed_unicode

        # Building the rect details of the character, based on its ascii code
        # and the table parameters
        index_diff = ascii_index - index_0
        char_col = index_diff % elsize + 1
        char_row = index_diff // elsize + 1
        char_x0 = x0 + (char_col - 1) * side
        char_y0 = y0 + (char_row - 1) * side
        char_rect = [char_x0, char_y0, char_x0 + side, char_y0 + side]

        # Finding all pen strokes in a particular square of the ascii table that are intersecting
        # with
        select_strokes_ = [x for x in ascii_ps_list if intersection_area(
            char_rect, [x['min_c_x'], x['min_c_y'], x['max_c_x'], x['max_c_y']], threshold=threshold)]

        # A deep copy is necessary here otherwise we risk altering the pen strokes in ascii_ps_list
        # if there is a transformation ratio
        select_strokes = copy.deepcopy(select_strokes_)

        # If either  ratio x or ratio y is given, enter transformation
        if (ratio_x != 1) or (ratio_y != 1):

            # Find starting point, assumed to be the min x and miny of all contours
            # The base reference for positioning letters is the bottom left corner (x-min, y_max)
            # This simplifies baseline  positionning for exceptions such as f, g, p, q, y
            try:
                start_x_std = min([stroke['min_c_x'] for stroke in select_strokes])
                start_y_std = min([stroke['max_c_y'] for stroke in select_strokes])
            except Exception as e:
                e
                # print()
                # print(f'*** char_from_ascii_tps - start_x_std : {e} - char : {searched_char} - ascii_index: {ascii_index}')
                return [], 0, 0, temp_dict

            # Compute the equivalent starting point in adb coordinates
            start_y_adb, start_x_adb = topleft_to_topright(
                [start_x_std, start_y_std], max_horizontal_pixels, max_vertical_pixels,
                adb_screen_max_x, adb_screen_max_y)

            if len(select_strokes) > 0:
                for an_el in select_strokes:
                    if 'vector_points' in an_el:
                        # Retrieve previous vector point
                        prev_vp = an_el['vector_points']
                        # Apply transformation ratio  relative to starting corner (start_y_adb, start_x_adb)
                        # and store in new vector point
                        # Such points are stores with top right coordinates (y_adb, x_adb)
                        new_vp = [
                            [
                                round(start_y_adb + (x[0] - start_y_adb) * ratio_y),
                                round(start_x_adb + (x[1] - start_x_adb) * ratio_x)
                                ] for x in prev_vp
                                ]

                        an_el['vector_points'] = new_vp

                        if 'contours' in an_el:
                            # Applying same transformation for contours points, this time
                            # using the equivalent top left origin starting point (start_x_std, start_y_std)
                            new_contours_dict = {}
                            prev_contour_dict = an_el['contours']
                            for contour_nb, prev_contour_vector in prev_contour_dict.items():
                                new_contour_vector = [
                                    [
                                        round(start_x_std + (x[0] - start_x_std) * ratio_x),
                                        round(start_y_std + (x[1] - start_y_std) * ratio_y)
                                        ] for x in prev_contour_vector
                                        ]
                                new_contours_dict[contour_nb] = new_contour_vector
                            # Updating the contours dictionary
                            an_el['contours'] = new_contours_dict

                        # Updating the contours statistics with the same transformation
                        min_c_x = an_el['min_c_x']
                        max_c_x = an_el['max_c_x']
                        min_c_y = an_el['min_c_y']
                        max_c_y = an_el['max_c_y']
                        avg_c_x = an_el['avg_c_x']
                        avg_c_y = an_el['avg_c_y']

                        new_min_c_x = round(start_x_std + (min_c_x - start_x_std) * ratio_x)
                        new_max_c_x = round(start_x_std + (max_c_x - start_x_std) * ratio_x)
                        new_min_c_y = round(start_y_std + (min_c_y - start_y_std) * ratio_y)
                        new_max_c_y = round(start_y_std + (max_c_y - start_y_std) * ratio_y)
                        new_avg_c_x = round(start_x_std + (avg_c_x - start_x_std) * ratio_x)
                        new_avg_c_y = round(start_y_std + (avg_c_y - start_y_std) * ratio_y)

                        an_el['min_c_x'] = new_min_c_x
                        an_el['max_c_x'] = new_max_c_x
                        an_el['min_c_y'] = new_min_c_y
                        an_el['max_c_y'] = new_max_c_y
                        an_el['avg_c_x'] = new_avg_c_x
                        an_el['avg_c_y'] = new_avg_c_y

        min_x_v = [x['min_c_x'] for x in select_strokes]
        max_x_v = [x['max_c_x'] for x in select_strokes]
        min_y_v = [x['min_c_y'] for x in select_strokes]
        max_y_v = [x['max_c_y'] for x in select_strokes]

        if min_x_v != [] and max_x_v != [] and min_y_v != [] and max_y_v != []:
            min_x = min(min_x_v)
            max_x = max(max_x_v)
            min_y = min(min_y_v)
            max_y = max(max_y_v)

            stroke_width = max_x - min_x
            stroke_height = max_y - min_y

            # Storing retrieved pen strokes of a given letter,
            # with a particular transformation, in the temporary dict
            temp_dict[a_key] = (select_strokes, stroke_width, stroke_height)

            return select_strokes, stroke_width, stroke_height, temp_dict

    except Exception as e:
        print()
        print(f'*** char_from_ascii_tps: {e} - char : {searched_char}')
        return [], 0, 0, temp_dict


def build_contours(paths_list, start_index=0):
    """ Takes a list of vector-points and returns a dictionary where each key
        is the index of the original list and the values are a dictionary with
        keys
        - 'path' for a vector point
        - 'contour' for the correstponding contour """
    try:
        paths_dict = {}
        path_index = start_index
        for a_path in paths_list:
            dc_index = -1
            try:
                # Create a LineString from the path points
                line = LineString(a_path[:-1])

                # Generate a buffer (offset contour) around the line
                buffer_polygon = line.buffer(0.1, resolution=CONTOUR_RESOLUTION, cap_style='flat', join_style='round')

                # TODO: Check if this is more appropriate: offset_polygon = line.offset_curve(0.1)

                try:
                    # Extract the exterior coordinates of the buffer polygon
                    # contour_points_ = list(buffer_polygon.exterior.coords)
                    # contour_points = [[round(x[0]), round(x[1])] for x in contour_points_]

                    # Using the "normal" contours
                    boundary = buffer_polygon.boundary
                    list_contours = []

                    # If it's a MultiLineString
                    if type(boundary) is MultiLineString:

                        contour_points = []
                        for geom in boundary.geoms:
                            points = list(geom.coords)
                            contour_points.extend([[round(x[0], 4), round(x[1], 4)] for x in points])
                            list_contours.append([[round(x[0], 4), round(x[1], 4)] for x in points])

                    else:
                        # If it's a single LineString
                        boundary_points = list(boundary.coords)

                        contour_points = [[round(x[0], 4), round(x[1], 4)] for x in boundary_points]
                        list_contours.append([[round(x[0], 4), round(x[1], 4)] for x in boundary_points])

                    # # Approximating the contour with the path itself
                    # # (this is not ideal but somehow works)
                    # contour_points = [[round(x[0]), round(x[1])] for x in a_path]

                    paths_dict[str(path_index)] = {
                        "path": a_path, "contours": contour_points,
                        "dc_index": dc_index, "list_contours": list_contours}
                    # print(f'====len contour_points: {len(contour_points)}  -- len list_contours: {len(list_contours[0])}')
                    # print(list_contours)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f'**- buffer_polygon: {e}')

                path_index += 1
            except Exception as e:
                if DEBUG_MODE:
                    print(f'**- buffer_polygon: {e}')
    except Exception as e:
        print(f'**- build_contours: {e}')
    return paths_dict


def vectors2psd(
    paths_dict, unstitch=UNSTITCH, pen_size=DEFAULT_PEN_SIZE,
        pen_color=DEFAULT_PEN_COLOR, series=NOTEBOOK_DEVICE):
    """ Build a basic pen_strokes_dict object from vector points and contour vectors"""

    try:
        pen_strokes_dict = {}
        used_rand = set()
        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)

        offset_x = OFFSET_T_X*max_horizontal_pixels/adb_screen_max_x
        offset_y = OFFSET_T_Y*max_vertical_pixels/adb_screen_max_y

        strokes = []
        strokes_nb = len(paths_dict.keys())

        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)
        for a_path_nb, a_path_bundle in paths_dict.items():

            # Intitialize and populate an individual stroke dict
            a_stroke_dict = {}
            a_stroke_dict_filling = {}

            # TODO: Keep this below
            list_contours = a_path_bundle['list_contours']

            contours_vector = a_path_bundle['contours']

            pen_color_n = HIGHLIGHTS_COLOR[pen_color][0]

            pen_weight_n = NEEDLE_POINT_SIZES[pen_size]

            if UNSTITCH:
                # dc_index = a_path_bundle['dc_index']
                # a_stroke_dict['contours'] = {"1": contours_vector}
                # I don't understand why the below doesn't work

                a_stroke_dict['contours'] = {str(index + 1): value for index, value in enumerate(list_contours) if len(list_contours) > 0}
                a_stroke_dict_filling['contours'] = {str(index + 1): value for index, value in enumerate(list_contours) if len(list_contours) > 0}

                x_coords, y_coords = zip(*contours_vector)

                # Calculate min, max, and averages
                min_x = round(min(x_coords) - offset_x)
                max_x = round(max(x_coords) - offset_x + 1)
                min_y = round(min(y_coords) - offset_y)
                max_y = round(max(y_coords) - offset_y + 1)
                avg_x = round(sum(x_coords) / len(x_coords) - 1)
                avg_y = round(sum(y_coords) / len(y_coords) - 1)
                a_stroke_dict['min_c_x'] = max(min(max_vertical_pixels - min_y, max_vertical_pixels), 0)
                a_stroke_dict['min_c_y'] = max(min(min_x, max_horizontal_pixels), 0)
                a_stroke_dict['max_c_x'] = max(min(max_vertical_pixels - max_y, max_vertical_pixels), 0)
                a_stroke_dict['max_c_y'] = max(min(max_x, max_horizontal_pixels), 0)
                a_stroke_dict['avg_c_x'] = max(min(max_vertical_pixels - avg_y, max_vertical_pixels), 0)
                a_stroke_dict['avg_c_y'] = max(min(avg_x, max_horizontal_pixels), 0)

            else:
                # dc_index = a_path_bundle['dc_index']
                a_stroke_dict['contours'] = {"1": contours_vector}
                a_stroke_dict_filling['contours'] = {"1": contours_vector}
                # I don't understand why the below doesn't work
                # a_stroke_dict['contours'] = {str(index + 1): value for index, value in enumerate(list_contours)}
                if len(contours_vector) > 0:
                    x_coords, y_coords = zip(*contours_vector)

                    # Calculate min, max, and averages
                    min_x = round(min(x_coords) - offset_x)
                    max_x = round(max(x_coords) - offset_x + 1)
                    min_y = round(min(y_coords) - offset_y)
                    max_y = round(max(y_coords) - offset_y + 1)
                    avg_x = round(sum(x_coords) / len(x_coords) - 1)
                    avg_y = round(sum(y_coords) / len(y_coords) - 1)
                else:
                    min_x = min_x = min_x = min_x = min_x = 0

                a_stroke_dict['min_c_x'] = max(min(min_x, max_horizontal_pixels), 0)
                a_stroke_dict['min_c_y'] = max(min(min_y, max_vertical_pixels), 0)
                a_stroke_dict['max_c_x'] = max(min(max_x, max_horizontal_pixels), 0)
                a_stroke_dict['max_c_y'] = max(min(max_y, max_vertical_pixels), 0)
                a_stroke_dict['avg_c_x'] = max(min(avg_x, max_horizontal_pixels), 0)
                a_stroke_dict['avg_c_y'] = max(min(avg_y, max_vertical_pixels), 0)

            screen_vector_points = a_path_bundle['path']

            # denormalized_points = [topleft_to_topright(x, series=series, unstitch_mode=unstitch, offset_h=-OFFSET_T_X, offset_v=OFFSET_T_Y) for x in screen_vector_points if len(x) > 0]
            denormalized_points = [
                topleft_to_topright(
                    x, max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x,
                    adb_screen_max_y, unstitch_mode=unstitch, offset_h=0, offset_v=0) for x in screen_vector_points if len(x) > 0]

            a_stroke_dict['vector_points'] = denormalized_points
            vector_size = len(denormalized_points)
            a_stroke_dict['vector_size'] = vector_size
            a_stroke_dict['type'] = DEFAULT_PEN
            a_stroke_dict['color'] = pen_color_n
            a_stroke_dict['weight'] = pen_weight_n
            a_stroke_dict['vector_pressure'] = [DEFAULT_PRESSURE] * vector_size
            a_stroke_dict['vector_one'] = [1] * vector_size

            draw_random = True

            while draw_random:
                rand_unsigned_int = random.randint(0, (2**32) - 1)
                draw_random = rand_unsigned_int in used_rand
                used_rand.add(rand_unsigned_int)

            # rand_unsigned_int = random.randint(0, (2**32) - 1)
            a_stroke_dict['vector_unique'] = [rand_unsigned_int] * vector_size
            r_bytes = 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000F03F'
            if series != 'N5':
                # r_bytes += '0000000000000000007C050000500700000000000000040000006E6F6E65040000006E6F6E650000000001000000'
                r_bytes += '0000000000000000007C050000500700000000000000040000006E6F6E65040000006E6F6E6500000000030000000200000000000000000000000000000000000000'
            else:
                # r_bytes += '00000000000000000080070000000A00000000000000040000006E6F6E65040000006E6F6E650000000001000000'
                r_bytes += '00000000000000000080070000000A00000000000000040000006E6F6E65040000006E6F6E6500000000030000000200000000000000000000000000000000000000'

            a_stroke_dict['r_bytes'] = r_bytes

            strokes.append(a_stroke_dict)

        pen_strokes_dict['strokes'] = strokes
        pen_strokes_dict['strokes_nb'] = strokes_nb

    except Exception as e:
        print(f'*** vectors2psd: {e}')
        return {}

    return pen_strokes_dict
