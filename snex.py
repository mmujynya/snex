import psyn_utils as pysn
import uuid
import time
import random
import sys

SCREEN_RATIO = 3

PAGE_SPACING = 50
HORIZONTAL_PAGES = False  # If True, the pages will be displayed horizontally
RESOLUTIONS = {
        "N6": (1404, 1872),
        "N5": (1920, 2560)
    }

INITIAL_X = 0
INITIAL_Y = 0
COLORS = {
    0: "#1e1e1e",
    157: "#9d9d9d",
    158: "#9d9d9d",
    201: "#c9c9c9",
    202: "#c9c9c9",
    254: "#ffffff",
    255: "#ffffff"
}
MYSCRIPT_RATIO = 12
VERSION = 1.0


def generate_excalidraw_id(use_uuid=True, length=20):
    """
    Generate a random string of the given length composed of characters
    commonly used in Excalidraw element IDs.
    """
    if use_uuid:
        return str(uuid.uuid4())
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def create_pages(
        series, num_pages, resolutions=RESOLUTIONS, horizontal_pages=HORIZONTAL_PAGES, spacing=PAGE_SPACING,
        initial_x=INITIAL_X, initial_y=INITIAL_Y, ocr_layer=False):
    """
    Create an Excalidraw file with num_pages frames, each representing a Supernote page.

    Args:
        series (str): Either "N6" or "N5" indicating the Supernote model.
        num_pages (int): The number of pages/frames to create.

    Returns:
        tuple: A tuple containing:
            - excalidraw_file (dict): The Excalidraw file as a dictionary.
            - page_mapping (dict): A mapping from page number (1-indexed) to the frame element.
    """
    # Check if the provided model is valid.
    if series not in resolutions:
        raise ValueError("Invalid model. Please use 'N6' or 'N5'.")

    frame_width, frame_height = resolutions[series]
    frame_width = int(frame_width/SCREEN_RATIO)
    frame_height = int(frame_height/SCREEN_RATIO)
    elements = []
    page_mapping = {}

    for i in range(num_pages):
        # Compute positions for horizontal or vertical arrangement.
        if horizontal_pages:
            x = initial_x + i * (frame_width + spacing)
            y = initial_y
        else:
            x = initial_x
            y = initial_y + i * (frame_height + spacing)

        # Create a frame element with additional properties to match the manual example.
        frame_id = generate_excalidraw_id()

        frame = {
            "id": frame_id,
            "type": "frame",
            "x": x,
            "y": y,
            "width": frame_width,
            "height": frame_height,
            "angle": 0,
            "strokeColor": "#bbb",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "index": f"a{i+3}",
            "roundness": None,
            "seed": random.randint(0, 1000000000),
            "version": random.randint(1, 1000),
            "versionNonce": random.randint(1, 1000000000),
            "isDeleted": False,
            "boundElements": [],
            "updated": int(time.time() * 1000),
            "link": None,
            "locked": False,
            "name": f"Page {i+1}"

        }
        elements.append(frame)
        if ocr_layer:
            ocr_frame = {
                "id": f'{frame_id}_ocr',
                "type": "frame",
                "x": x,
                "y": y,
                "width": frame_width,
                "height": frame_height,
                "angle": 0,
                "strokeColor": "#bbb",
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": "solid",
                "roughness": 0,
                "opacity": 0,
                "groupIds": [],
                "frameId": None,
                "index": f"aocr{i+3}",
                "roundness": None,
                "seed": random.randint(0, 1000000000),
                "version": random.randint(1, 1000),
                "versionNonce": random.randint(1, 1000000000),
                "isDeleted": False,
                "boundElements": [],
                "updated": int(time.time() * 1000),
                "link": None,
                "locked": False,
                "name": ""}
            elements.append(ocr_frame)
        page_mapping[i + 1] = frame

    # Build the overall Excalidraw file structure.
    excalidraw_file = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": 20,
            "gridStep": 5,
            "gridModeEnabled": False,
            "viewBackgroundColor": "#ffffff"
        },
        "files": {}
    }

    return excalidraw_file, page_mapping


def add_freedraw_to_page(
        color, vector_points, page_number, excalidraw_file, min_c_x, min_c_y, series, pen_stroke_width=0.3,
        screen_ratio=SCREEN_RATIO):
    """
    Add a freedraw element (hand-drawn stroke) to a specific page (frame) in the Excalidraw file.

    Args:
        color (str): Stroke color (e.g., "#1e1e1e").
        vector_points (list): A list of points (each a list or tuple of [x, y]) representing the stroke.
        page_number (int): The page number (as set in the frame's "name", e.g., "Page 1").
        excalidraw_file (dict): The Excalidraw file structure (as returned by create_pages).

    """
    try:
        # Find the frame corresponding to the given page number.
        vector_points = [(int((x[0]-min_c_x)/screen_ratio), int((x[1]-min_c_y)/screen_ratio)) for x in vector_points]

        target_frame = None
        for el in excalidraw_file.get("elements", []):
            if el.get("type") == "frame" and el.get("name") == f"Page {page_number}":
                target_frame = el
                break
        if target_frame is None:
            raise ValueError(f"Frame for Page {page_number} not found.")

        # Set the drawing's position relative to the frame.
        base_width = target_frame.get("width", 0)
        base_height = target_frame.get("height", 0)

        # Frame Positionning
        denominator_x = RESOLUTIONS[series][0]
        denominator_y = RESOLUTIONS[series][1]

        if HORIZONTAL_PAGES:
            freedraw_x = (int(page_number) - 1)*(PAGE_SPACING + base_width) + min_c_x + base_width * min_c_x / denominator_x
            freedraw_y = min_c_y+base_height*min_c_y/denominator_y
        else:
            freedraw_x = min_c_x+base_width*min_c_x/denominator_x
            freedraw_y = (int(page_number)-1)*(PAGE_SPACING+base_height)+min_c_y+base_height*min_c_y/denominator_y

        # Compute the bounding box of the provided vector points.
        if vector_points:
            min_x = min(pt[0] for pt in vector_points)
            max_x = max(pt[0] for pt in vector_points)
            min_y = min(pt[1] for pt in vector_points)
            max_y = max(pt[1] for pt in vector_points)
        else:
            min_x = max_x = min_y = max_y = 0

        width = max_x - min_x
        height = max_y - min_y

        vector_points = [(x[0]-min_c_x, x[1]-min_c_y) for x in vector_points]

        # Remove adjacent duplicates using list comprehension and enumerate
        vector_points = [pt for i, pt in enumerate(vector_points) if i == 0 or pt != vector_points[i-1]]

        # Create the freedraw element.
        freedraw_element = {
            "id": generate_excalidraw_id(),
            "type": "freedraw",
            "x": freedraw_x,
            "y": freedraw_y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": pen_stroke_width,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": target_frame["id"],
            "index": f"a{random.randint(1, 1000)}",  # Arbitrary index for ordering.
            "roundness": None,
            "seed": random.randint(0, 1000000000),
            "version": random.randint(1, 1000),
            "versionNonce": random.randint(0, 1000000000),
            "isDeleted": False,
            "boundElements": None,
            "updated": int(time.time() * 1000),
            "link": None,
            "locked": False,
            "points": vector_points,
            "pressures": [],
            "simulatePressure": True,
            "lastCommittedPoint": vector_points[-1] if vector_points else None,
        }

        # Append the new freedraw element to the Excalidraw file.
        excalidraw_file["elements"].append(freedraw_element)
    except Exception as e:
        print(f'**-add_freedraw_to_page: {e}')


def add_ocr_to_page(
        page_number, excalidraw_file, word_rect_std, word_text, series, screen_ratio=SCREEN_RATIO):
    """
    Add ocr to an ocr frame pertaining to a page.

    """
    # Find the frame corresponding to the given page number.
    word_rect_std = [int(x/screen_ratio) for x in word_rect_std]

    min_c_x, min_c_y, width, height = word_rect_std

    target_frame = None
    id_target_frame = ''
    for el in excalidraw_file.get("elements", []):
        if el.get("type") == "frame" and el.get("name") == f"Page {page_number}":
            id_target_frame = f'{el.get("id")}_ocr'
            break
    if id_target_frame != '':
        for el in excalidraw_file.get("elements", []):
            if el.get("type") == "frame" and el.get("id") == id_target_frame:
                target_frame = el
                break

    if target_frame is None:
        raise ValueError(f"OCR frame for Page {page_number} not found.")

    # Set the drawing's position relative to the frame.
    base_width = target_frame.get("width", 0)
    base_height = target_frame.get("height", 0)

    if HORIZONTAL_PAGES:
        text_x = (int(page_number) - 1)*(PAGE_SPACING + base_width) + min_c_x
        text_y = min_c_y
    else:
        text_x = min_c_x
        text_y = (int(page_number)-1)*(PAGE_SPACING+base_height) + min_c_y

    # Create the text element.
    text_element = {
        "id": generate_excalidraw_id(),
        "type": "text",
        "x": text_x,
        "y": text_y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 0.3,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": id_target_frame,
        "index": f"a{random.randint(1, 1000)}",  # Arbitrary index for ordering.
        "roundness": None,
        "seed": random.randint(0, 1000000000),
        "version": random.randint(1, 1000),
        "versionNonce": random.randint(0, 1000000000),
        "isDeleted": False,
        "boundElements": None,
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
        "text": word_text,
        "fontSize": 28,
        "fontFamily": 5,
        "textAlign": "left",
        "verticalAlign": "top",
        "containerId": None,
        "originalText": word_text,
        "autoResize": True,
        "lineHeight": 1.25
    }

    # Append the new text element to the Excalidraw file.
    excalidraw_file["elements"].append(text_element)


def note2ex(note_fn, series):
    """
    Input:
        - filename of a Supernote notebook
        - series of the Supernote device that created the notebook
          ('N5' for Manta, 'N6' for Nomad)
    Output:
        - An Excalidraw file
    """
    try:
        print(f'Processing file: {note_fn} with series: {series}')

        pen_strokes_dict, _, meta_data = pysn.get_pen_strokes_dict(note_fn)

        _, ocr_pages_dict = pysn.titles_and_text_from_notes(meta_data, note_fn)

        ocr_file = ocr_pages_dict != {}

        page_nb = len(pen_strokes_dict.keys())

        exca_file, _ = create_pages(series, page_nb, ocr_layer=ocr_file)

        pen_stroke_width = 0.3
        # Parse the dictionary of totalpath objects

        for apage, avalue in pen_strokes_dict.items():

            a_list_strokes = avalue['strokes']

            for a_stroke in a_list_strokes:

                min_c_x = a_stroke['min_c_x']
                min_c_y = a_stroke['min_c_y']

                try:
                    a_stroke_color = COLORS[a_stroke['color']]
                except Exception as e:
                    print(f'**- Color unknown: {e}')
                    a_stroke_color = "#9d9d9d",

                a_stroke_weight = a_stroke['weight']
                if a_stroke_weight <= 500:
                    pen_stroke_width = 0.3
                elif a_stroke_weight <= 1000:
                    pen_stroke_width = 1
                else:
                    pen_stroke_width = 2

                a_stroke_vector_points = a_stroke['vector_points']
                normalized_points = [
                    pysn.topright_to_topleft(x, series) for x in a_stroke_vector_points]
                add_freedraw_to_page(a_stroke_color, normalized_points, apage, exca_file, min_c_x, min_c_y,
                                     series, pen_stroke_width=pen_stroke_width)

            if ocr_file:
                page_ocr = str(int(apage)-1)
                if page_ocr in ocr_pages_dict:
                    ocr_page = ocr_pages_dict[page_ocr]
                    if 'elements' in ocr_page:
                        elements_list = ocr_page['elements']
                        for an_element in elements_list:
                            if 'words' in an_element:
                                words_list = an_element['words']
                                for a_word in words_list:
                                    if 'bounding-box' in a_word:
                                        b_b = a_word['bounding-box']
                                        word_rect = [
                                            b_b['x'], b_b['y'],
                                            b_b['width'],
                                            b_b['height']]
                                        word_rect_std = [x*MYSCRIPT_RATIO for x in word_rect]
                                        word_text = a_word['label']
                                        if word_text != '':
                                            add_ocr_to_page(apage, exca_file, word_rect_std, word_text, series)

        output_fn = f'{note_fn[:-5]}.excalidraw'
        pysn.save_json(output_fn, exca_file)
        print(f'Generated file: {output_fn}')
    except Exception as e:
        print(f'*** note2ex: {e}')
        print(ocr_file)
        print(page_nb)
        print(f'---apage:{apage}')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python snex.py <filename> <series>")
        sys.exit(1)
    print()
    print(f'SNEX Version {VERSION}')
    print('----------------')
    filename = sys.argv[1]
    series = sys.argv[2]

    note2ex(filename, series)
