import json
import os
import supernotelib as sn
import base64

OFFSET_T_X = 16
OFFSET_T_Y = 32
MAX_HORIZONTAL_PIXELS_N6 = 1404        # Default pixels size on A6x2. You may have to change that for other devices
MAX_VERTICAL_PIXELS_N6 = 1872          # Default pixels size on A6x2. You may have to change that for other devices
MAX_HORIZONTAL_PIXELS_N5 = 1920
MAX_VERTICAL_PIXELS_N5 = 2560
ADB_SCREEN_MAX_X_N6 = 11864    # Inferred from binaries;I previously estimated this to be 11800
ADB_SCREEN_MAX_X_N5 = 16224   # For N5 series like Manta
ADB_SCREEN_MAX_Y_N6 = 15819    # I previously estimated this to be 15800
ADB_SCREEN_MAX_Y_N5 = 21632  # For N5 series like Manta
ADB_SCREEN_OFFSET = 5


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
    return contains_hex_sequence(binary_data, '7C0500005007')


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


def topright_to_topleft(a_reversed_point, series):
    """ Screen coordinates conversion
        Warning: reverts back to x & y """
    try:
        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)
        a_point = (
            max(min(
                round((adb_screen_max_x+OFFSET_T_X-a_reversed_point[1])*max_horizontal_pixels/adb_screen_max_x), max_horizontal_pixels), 0),
            max(min(round((a_reversed_point[0]+OFFSET_T_Y)*max_vertical_pixels/adb_screen_max_y), max_vertical_pixels), 0)
        )
    except Exception as e:
        print(f'**-topright_to_topleft:{e}')
    return a_point


def topleft_to_topright(a_point, series, unstitch_mode=False, offset_h=0, offset_v=0):
    """ Screen coordinates conversion Warning: reverses x & y """
    try:
        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = series_bounds(series)
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
