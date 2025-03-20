import psyn_utils as pysn
import uuid
import time
import random
import sys
import re
import json
import os

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
VERSION = 1.02


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
            "strokeWidth": 0.3,
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
            "locked": True,
            "name": f"Page {i+1}",
            "pysn": series

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
                "strokeWidth": 0.3,
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
                "locked": True,
                "name": "",
                "pysn": series}
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


def page_number2frame(excalidraw_file, page_number, ocr=False):
    """ Returns a frame from a page number """

    target_frame = None
    target_frame_ocr = None

    # Find the page frame
    for el in excalidraw_file.get("elements", []):
        if el.get("type") == "frame" and el.get("name") == f"Page {page_number}":
            target_frame = el
            break

    # Find the corresponding ocr frame
    if target_frame is not None:
        if ocr:
            id_target_frame = f'{target_frame.get("id")}_ocr'

            for el in excalidraw_file.get("elements", []):
                if el.get("type") == "frame" and el.get("id") == id_target_frame:
                    target_frame_ocr = el
                    break

    return target_frame, target_frame_ocr


def add_ps2canvas(
        pen_stroke, target_frame, page_number, excalidraw_file, series,
        screen_ratio=SCREEN_RATIO):
    """
    Add a freedraw element (hand-drawn stroke) to a specific page (frame) in the Excalidraw file from
    a pen stroke dict.

    Args:
        page_number (int): The page number (as set in the frame's "name", e.g., "Page 1").
        excalidraw_file (dict): The Excalidraw file structure (as returned by create_pages).

    """
    try:
        # retrieve the color
        try:
            color = COLORS[pen_stroke['color']]
        except Exception as e:
            print(f'**- Color unknown: {e}')
            color = "#9d9d9d"

        # Retrieve the pen weight (pen_stroke_width)
        stroke_weight = pen_stroke['weight']

        if stroke_weight <= 500:
            pen_stroke_width = 0.3
        elif stroke_weight <= 1000:
            pen_stroke_width = 1
        else:
            pen_stroke_width = 2

        min_c_x = pen_stroke['min_c_x']
        min_c_y = pen_stroke['min_c_y']

        # Retrieve the vector_points
        stroke_vector_points = pen_stroke['vector_points']
        # print(f'---min_c_x: {min_c_x} - min_c_y: {min_c_y}')
        # print(f'---stroke_vector_points: {stroke_vector_points}')
        normalized_points = [pysn.topright_to_topleft(x, series) for x in stroke_vector_points]
        # print(f'---normalized_points: {normalized_points}')

        # Find the frame corresponding to the given page number.
        vector_points = [(int((x[0]-min_c_x)/screen_ratio), int((x[1]-min_c_y)/screen_ratio)) for x in normalized_points]

        # Set the drawing's position relative to the frame.
        base_width = target_frame.get("width", 0)
        base_height = target_frame.get("height", 0)

        # Frame Positionning
        denominator_x = RESOLUTIONS[series][0]
        denominator_y = RESOLUTIONS[series][1]

        if HORIZONTAL_PAGES:
            freedraw_x = (page_number - 1)*(PAGE_SPACING + base_width) + min_c_x + base_width * min_c_x / denominator_x
            freedraw_y = min_c_y+base_height*min_c_y/denominator_y
        else:
            freedraw_x = min_c_x+base_width*min_c_x/denominator_x
            freedraw_y = (page_number-1)*(PAGE_SPACING+base_height)+min_c_y+base_height*min_c_y/denominator_y

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
        print(f'**-add_ps2canvas: {e}')


def add_text2canvas(target_frame, page_number, excalidraw_file, word_rect_std, word_text, screen_ratio=SCREEN_RATIO):
    """
    Add ocr to an ocr frame pertaining to a page.

    """

    # Find the frame corresponding to the given page number.
    word_rect_std = [int(x/screen_ratio) for x in word_rect_std]

    min_c_x, min_c_y, width, height = word_rect_std

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
        "opacity": 0,
        "groupIds": [],
        "frameId": target_frame.get("id", ""),
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
        "fontSize": 16,
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


def note2ex(note_fn):
    """
    Input:
        - filename of a Supernote notebook
        - series of the Supernote device that created the notebook
          ('N5' for Manta, 'N6' for Nomad)
    Output:
        - An Excalidraw file
    """
    try:
        print(f'Processing file: {note_fn}')

        pen_strokes_dict, _, meta_data, series = pysn.get_pen_strokes_dict(note_fn)

        _, ocr_pages_dict = pysn.titles_and_text_from_notes(meta_data, note_fn)

        ocr_file = ocr_pages_dict != {}

        page_nb = len(pen_strokes_dict.keys())

        exca_file, _ = create_pages(series, page_nb, ocr_layer=ocr_file)

        # Parse the dictionary of totalpath objects

        for apage, avalue in pen_strokes_dict.items():

            target_frame, target_frame_ocr = page_number2frame(exca_file, apage, ocr=ocr_file)

            page_number = int(apage)

            a_list_strokes = avalue['strokes']

            for a_stroke in a_list_strokes:
                add_ps2canvas(
                    a_stroke, target_frame, page_number, exca_file, series)

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
                                            add_text2canvas(target_frame_ocr, page_number, exca_file, word_rect_std, word_text)

        output_fn = f'{note_fn[:-5]}.excalidraw'
        pysn.save_json(output_fn, exca_file)
        print(f'Generated file: {output_fn}')
    except Exception as e:
        print(f'*** note2ex: {e}')
        print(ocr_file)


def extract_penstrokes_by_page(excalidraw_filename, screen_ratio=SCREEN_RATIO):
    """
    Load an Excalidraw file from disk and extract pen stroke data for each page.
    Each page key maps to a list of tuples: (strokeColor, strokeWidth, vector_points),
    where vector_points are recovered relative to the frame's coordinate system.

    Args:
        excalidraw_filename (str): Path to the Excalidraw JSON file.
        series: Identifier used for resolution lookup (RESOLUTIONS[series] should be available).
        screen_ratio (float): The screen ratio used during the forward transformation.

    Returns:
        dict: Keys are page numbers (from frame names like "Page 1"), and values are lists of tuples.
              Each tuple contains (strokeColor, strokeWidth, vector_points).
    """
    # Load the Excalidraw file.
    with open(excalidraw_filename, 'r') as f:
        excalidraw_file = json.load(f)

    # Build a mapping from frame id to frame properties (including x, y positions).
    frame_mapping = {}
    series = excalidraw_file.get("pysn", "N5")
    for elem in excalidraw_file.get("elements", []):
        if elem.get("type") == "frame":
            name = elem.get("name", "")
            match = re.search(r'\bPage\s+(\d+)\b', name)
            if match:
                page_number = int(match.group(1))
                frame_mapping[elem.get("id")] = {
                    "page_number": page_number,
                    "width": elem.get("width", 0),
                    "height": elem.get("height", 0),
                    "x": elem.get("x", 0),  # Record the frame's x position.
                    "y": elem.get("y", 0),  # Record the frame's y position.
                }

    # Prepare dictionary to collect pen strokes by page.
    page_penstrokes = {}

    # Process freedraw elements.
    for elem in excalidraw_file.get("elements", []):
        if elem.get("type") != "freedraw":
            continue

        frame_id = elem.get("frameId")
        if frame_id not in frame_mapping:
            # Freedraw element not attached to a recognized frame.
            continue

        # Retrieve frame parameters.
        frame_props = frame_mapping[frame_id]
        page_number = frame_props["page_number"]
        base_width = frame_props["width"]
        base_height = frame_props["height"]
        frame_x = frame_props["x"]
        frame_y = frame_props["y"]
        denominator_x, denominator_y = RESOLUTIONS[series]

        # First, compute the freedraw element's position relative to the frame.
        # This step subtracts out the frame's absolute x,y position.
        relative_x = elem["x"] - frame_x
        relative_y = elem["y"] - frame_y

        # Invert the frame positioning to recover min_c_x and min_c_y.
        if HORIZONTAL_PAGES:
            # For horizontal pages, account for the page offset in x.
            min_c_x = (relative_x - (page_number - 1) * (PAGE_SPACING + base_width)) / (1 + base_width / denominator_x)
            min_c_y = relative_y / (1 + base_height / denominator_y)
        else:
            min_c_x = relative_x / (1 + base_width / denominator_x)
            min_c_y = (relative_y - (page_number - 1) * (PAGE_SPACING + base_height)) / (1 + base_height / denominator_y)

        # Invert the vector point transformation.
        # The forward transformation roughly did:
        #   temp_point = int((normalized - min_c) / screen_ratio)
        #   stored_point = temp_point - min_c
        # Here we add back min_c and reverse the scaling.
        recovered_points = []
        for pt in elem.get("points", []):
            temp_x = pt[0] + min_c_x
            temp_y = pt[1] + min_c_y
            normalized_x = temp_x * screen_ratio + min_c_x
            normalized_y = temp_y * screen_ratio + min_c_y
            denormalized_point = pysn.topleft_to_topright((normalized_x, normalized_y), series)
            recovered_points.append(denormalized_point)

        # Build the tuple (strokeColor, strokeWidth, vector_points) for this pen stroke.
        stroke_tuple = (elem.get("strokeColor"), elem.get("strokeWidth"), recovered_points)
        page_penstrokes.setdefault(page_number, []).append(stroke_tuple)

    return page_penstrokes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python snex.py <filename>")
        sys.exit(1)
    print()
    print(f'SNEX Version {VERSION}')
    print('-----------------')
    filename = sys.argv[1]
    # Get the file extension
    basename, file_extension = os.path.splitext(filename)

    if file_extension.lower() == '.note':
        note2ex(filename)
    elif file_extension.lower() == '.excalidraw':
        adict = extract_penstrokes_by_page(filename)
        output_fn = f'{basename}.json'
        pysn.save_json(output_fn, adict)
        print(f'Generated file: {output_fn}')
