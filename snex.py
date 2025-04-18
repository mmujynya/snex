import psyn_utils as pysn
import uuid
import time
import random
import sys
import re
import json
import os
import math
import argparse
import base64
from io import BytesIO
from PIL import Image
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
import numpy as np

unique_list = []
update_list = []
VERSION = 1.06


def resize_strokes(strokes, target_width, target_height, dest_origin=(0, 0), preserve_aspect_ratio=True):
    """
    Resize a list of strokes (each stroke is a list of (x, y) points) so that the
    overall content fits within the target width and height. Then translate the
    strokes so that the bounding box’s top-left corner moves to the specified destination.

    Parameters:
        strokes (list of list of tuples): The strokes to resize.
        target_width (float): The target width for scaling.
        target_height (float): The target height for scaling.
        dest_origin (tuple): (x, y) where you want the top-left corner of the resized
                             image to be located.
        preserve_aspect_ratio (bool): If True, scales uniformly to preserve the aspect ratio.
                                      If False, scales x and y independently.

    Returns:
        list of list of tuples: The resized and translated strokes.
    """
    # Flatten all points to determine the original bounding box.
    all_points = [pt for stroke in strokes for pt in stroke]
    xs = [pt[0] for pt in all_points]
    ys = [pt[1] for pt in all_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    original_width = max_x - min_x
    original_height = max_y - min_y
    if original_width == 0 or original_height == 0:
        raise ValueError("Strokes have zero width or height; cannot be resized.")

    new_strokes = []

    if preserve_aspect_ratio:
        # Compute uniform scale factor based on the limiting dimension.
        scale = min(target_width / original_width, target_height / original_height)

        # Use dest_origin as the new top-left corner.
        offset_x = dest_origin[0]
        offset_y = dest_origin[1]

        # If you wish, you can still center the image inside the target dimensions and then
        # move it to dest_origin, but here we assume that dest_origin is where the scaled image's
        # top-left corner should be placed.
        for stroke in strokes:
            new_stroke = []
            for (x, y) in stroke:
                new_x = (x - min_x) * scale + offset_x
                new_y = (y - min_y) * scale + offset_y
                new_stroke.append((new_x, new_y))
            new_strokes.append(new_stroke)
    else:
        # Independent scaling for x and y.
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        offset_x = dest_origin[0]
        offset_y = dest_origin[1]

        for stroke in strokes:
            new_stroke = []
            for (x, y) in stroke:
                new_x = (x - min_x) * scale_x + offset_x
                new_y = (y - min_y) * scale_y + offset_y
                new_stroke.append((new_x, new_y))
            new_strokes.append(new_stroke)

    return new_strokes


def svg_to_relative_strokes(origin, svg_content, num_samples=pysn.XC_NUM_SAMPLES, tolerance=1e-9):
    """
    Extracts separate strokes (continuous vector subpaths) from an SVG image and
    converts the points to coordinates relative to a given origin (top-left).

    Instead of using only segment endpoints, this version samples additional
    points along each segment (using 'num_samples'). This leads to smoother curves.

    Parameters:
        origin (tuple): (ox, oy) representing the top-left origin.
        svg_content (str or bytes): The SVG content. If provided as bytes, it will be decoded.
        num_samples (int): Number of sample points per segment (including endpoints).
                           Increasing this number increases smoothness.
        tolerance (float): Numerical tolerance for comparing points.

    Returns:
        list of list of tuples: Each inner list represents one stroke (list of (x, y) points
        relative to the origin). Separate strokes are created for discontinuous parts.
    """
    ox, oy = origin

    # If the SVG content is given as bytes, decode it.
    if isinstance(svg_content, bytes):
        svg_content = svg_content.decode('utf-8', errors='replace')

    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError as e:
        raise ValueError("Invalid SVG content provided.") from e

    # Define common namespace mappings (common with Inkscape or other SVG tools)
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
        'xlink': 'http://www.w3.org/1999/xlink'
    }

    # Look for <path> elements; try namespaced search first, then fallback.
    path_elements = root.findall('.//svg:path', ns)
    if not path_elements:
        path_elements = root.findall('.//path')

    all_strokes = []

    for path_elem in path_elements:
        d = path_elem.attrib.get('d')
        if not d:
            continue  # Skip if no path data.

        # Parse the path using svgpathtools.
        path_obj = parse_path(d)

        strokes = []  # List to hold strokes for this <path> element.
        current_stroke = []  # Points for the current stroke.

        for segment in path_obj:
            # Determine if we need to start a new stroke if there's a discontinuity.
            if current_stroke:
                # Check if the start of the segment matches the last point of the current stroke.
                if abs(segment.start - current_stroke[-1]) > tolerance:
                    strokes.append(current_stroke)
                    current_stroke = []

            # Sample points along this segment.
            ts = np.linspace(0, 1, num_samples)
            sampled_points = [segment.point(t) for t in ts]

            # If this segment continues an existing stroke, avoid duplicating the first point.
            if current_stroke:
                sampled_points = sampled_points[1:]

            # Extend the current stroke with the sampled points.
            current_stroke.extend(sampled_points)

        # Append any remaining stroke.
        if current_stroke:
            strokes.append(current_stroke)

        # Convert complex numbers to (x, y) tuples with coordinates relative to origin.
        for stroke in strokes:
            relative_stroke = [(pt.real + ox, pt.imag + oy) for pt in stroke]
            all_strokes.append(relative_stroke)

    return all_strokes


def load_excalidraw_images(filename, frame_mapping, series, screen_ratio, adict):
    """
    Loads all image elements from an Excalidraw file.

    Returns a dictionary with the RLE encoded image where the image was pasted
    """
    try:
        with open(filename, 'r', encoding='utf-8', errors="replace") as f:
            excalidraw_file = json.load(f)

        templates_dict = {}
        templates_dict_f = {}
        image_elements = [elem for elem in excalidraw_file.get("elements", []) if elem.get("type") == "image"]
        if not image_elements:
            raise ValueError("No image elements found in the Excalidraw file.")

        thick_ratio = globals()['pysn.XC_SVG_THIN_RATIO']

        for elem in image_elements:
            x = int(elem.get("x", 0))
            y = int(elem.get("y", 0))

            width = int(elem.get("width", 0))
            height = int(elem.get("height", 0))

            asset_id = elem.get("fileId")

            # Get the base64 image data from the "files" dictionary
            file_data = excalidraw_file.get("files", {}).get(asset_id, {})

            image_data = file_data.get("dataURL")
            if not image_data:
                print(f"Warning: Image data not found for assetId: {asset_id}. Skipping this element.")
                continue

            frame_id = elem.get("frameId")
            if frame_id not in frame_mapping:
                # Skip text elements not attached to a recognized frame.
                continue
            image_type = file_data.get("mimeType")

            # Retrieve frame parameters.
            frame_props = frame_mapping[frame_id]
            page_number = frame_props["page_number"]
            base_width = frame_props["width"]
            base_height = frame_props["height"]
            frame_x = frame_props["x"]
            frame_y = frame_props["y"]

            # Add advertisement pen-strokes (templates are not displaying for empty pages)
            if page_number not in adict:
                adict[page_number] = [
                        {'type': 'text',
                         'x': 1370,
                         'y': 120,
                         'width': 425,
                         'height': 45,
                         'color': '#1e1e1e',
                         'text': f'Generated by SNEX version {VERSION}',
                         'fontSize': 200,
                         'originalFontSize': 18,
                         'font_ratio': 0.55,
                         'fontFamily': 'barlow'}]

            # Remove the header if present
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            print(f'       - image_type: {image_type}')

            # Handling PNG
            if image_type in ['image/png', 'image/jpeg']:

                # Decode the image data and open it as a Pillow image

                pil_image = Image.open(BytesIO(image_bytes))
                # Resize to the specified dimensions
                pil_image = pil_image.resize((width*screen_ratio, height*screen_ratio))

                # if there is already a template in the dictionary, use it
                if page_number in templates_dict:
                    a_template_dict = templates_dict[page_number]
                    a_template = a_template_dict['image']
                else:
                    if series == 'N5':
                        a_template_fn = pysn.BLANK_TEMPLATE_PICTURE2
                    else:
                        a_template_fn = pysn.BLANK_TEMPLATE_PICTURE
                    a_template = Image.open(a_template_fn)
                    a_template_dict = {}

                if pysn.XC_HORIZONTAL_PAGES:
                    recovered_min_c_x = screen_ratio*(x - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_width))
                    recovered_min_c_y = screen_ratio*y
                else:
                    recovered_min_c_x = screen_ratio*x
                    recovered_min_c_y = screen_ratio*(y - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_height))
                a_template.paste(pil_image, (recovered_min_c_x, recovered_min_c_y))
                a_template_dict['image'] = a_template
                templates_dict[page_number] = a_template_dict

            # Handling SVG files
            elif image_type == 'image/svg+xml':
                print("       Tip: import a PNG image if you don't need to resize or edit ...")

                # Extract relative vector paths.
                # origin = (x, y)
                origin = (x-frame_x, y-frame_y)
                paths = svg_to_relative_strokes(origin, image_bytes)
                len_paths = len(paths)

                if len_paths < 100:
                    stwidth = 2*thick_ratio
                elif len_paths < 500:
                    stwidth = 1*thick_ratio
                else:
                    stwidth = 0.1*thick_ratio
                if len_paths > 0:

                    paths = resize_strokes(paths, width*screen_ratio, height*screen_ratio, dest_origin=origin)

                    for path in paths:
                        l_points = [[pt[0], pt[1]] for pt in path]
                        recovered_points = [pt for i, pt in enumerate(l_points) if i == 0 or pt != l_points[i-1]]

                        an_svg_image = {
                            'type': 'freedraw',
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'strokeWidth': stwidth,
                            'color': '#1e1e1e',
                            'points': recovered_points}
                        if page_number in adict:
                            dict_page = adict[page_number]
                            dict_page.append(an_svg_image)
                            adict[page_number] = dict_page
                        else:
                            adict[page_number] = [an_svg_image]

                # print("Relative vector paths:")
                # print(len(paths))
                # for i, path in enumerate(paths):
                #     print(f"Path {i+1}: len:{len(path)} {path[:5]}")

        for a_page, pil_image_dict in templates_dict.items():
            pil_image = pil_image_dict['image']
            png_buffer = BytesIO()
            pil_image.save(png_buffer, format='PNG')
            png_binary = png_buffer.getvalue()
            pil_image = pil_image.convert('L')
            pil_image_encoded, encoded = pysn.rle_encode_img(pil_image)

            if encoded:
                an_md5 = pysn.binary_md5(pil_image_encoded)
            else:
                pil_image_encoded = None
                an_md5 = 0

            # # Write the PNG binary content to a file
            # output_png_fn = os.path.join(source_folder, f'user_{an_md5}.png')

            # with open(output_png_fn, 'wb') as f:
            #     f.write(png_binary)

            pil_image_encoded = png_binary   # MMB testing
            templates_dict_f[str(a_page)] = {
                "image": pil_image,
                "rle_image": pil_image_encoded,
                "md5": an_md5}

        return adict, templates_dict_f
    except Exception as e:
        print(f'      - load_excalidraw_images: {e}')
        return adict, {}


def densify_stroke(stroke, factor=2, dashed=False, dotted=False, arrows=False):
    """
    Given a stroke (a list of [x, y] points), returns a new set of strokes with additional interim
    points inserted between consecutive points using linear interpolation. The 'factor' parameter
    specifies the number of subdivisions per segment.

    Parameters:
      stroke  : List of points (each point is a two-element list [x, y]).
      factor  : Integer; number of subdivisions to insert between each pair of original points.
      dashed  : Boolean; if True, the output is split into dash segments.
      dotted  : Boolean; if True, the output is reduced to isolated points (dots).

    Returns:
      A list of strokes. Each stroke is itself a list of [x, y] points. Under no circumstances
      will any stroke have fewer than 2 points.

    Raises:
      ValueError if both dashed and dotted are True or if factor is less than 1.
    """

    if dashed:
        dash_length = pysn.XC_DASH_LENGTH  # number of points for each dash segment.
        gap_length = pysn.XC_DASH_GAP_LENGTH    # number of points to skip.
    elif dotted:
        dash_length = pysn.XC_DOT_LENGTH  # number of points for each dot segment.
        gap_length = pysn.XC_DOT_GAP_LENGTH    # number of points to skip.

    if factor < 1:
        raise ValueError("factor must be >= 1")

    # Ensure the input stroke has at least 2 points.
    n = len(stroke)
    if n == 0:
        raise ValueError("Input stroke must contain at least one point.")
    if n == 1:
        stroke = [stroke[0], stroke[0]]
        n = 2

    # Step 1: Densify the stroke by linear interpolation.
    densified = []
    for i in range(n - 1):
        p0 = stroke[i]
        p1 = stroke[i + 1]
        # Insert 'factor' subdivisions on the segment from p0 to p1.
        for j in range(factor):
            u = j / factor
            x = (1 - u) * p0[0] + u * p1[0]
            y = (1 - u) * p0[1] + u * p1[1]
            densified.append([x, y])
    densified.append(stroke[-1])

    # Ensure densified stroke has at least 2 points.
    if len(densified) < 2:
        densified.append(densified[-1])

    # If neither dashed nor dotted is requested, return the full densified stroke.
    if not dashed and not dotted:
        return [densified]

    # Step 2: Apply dashed or dotted effects.
    if dashed or dotted:
        # For dashed lines we define a repeating pattern.
        # dash_length is the number of consecutive points to draw,
        # and gap_length is the number of points to skip.

        if arrows:
            dash_length = dash_length // 2
            gap_length = gap_length // 2

        pattern_length = dash_length + gap_length
        dashed_strokes = []
        current_dash = []
        for idx, pt in enumerate(densified):
            if (idx % pattern_length) < dash_length:
                current_dash.append(pt)
            else:
                if current_dash:
                    # Ensure the segment has at least 2 points.
                    if len(current_dash) == 1:
                        current_dash.append(current_dash[0])
                    dashed_strokes.append(current_dash)
                    current_dash = []
        if current_dash:
            if len(current_dash) == 1:
                current_dash.append(current_dash[0])
            dashed_strokes.append(current_dash)
        return dashed_strokes


def get_arrow_points(base_x, base_y, start_rel, end_rel, arrowhead_length=60, arrowhead_width=40,
                     middle_points_list=[], round=False):
    """
    Compute pen strokes for drawing an arrow that may have one or more elbows.

    The arrow is composed of:
      - An arrow shaft defined by a polyline that starts at the starting point,
        goes through any provided middle points (elbows), and ends at the arrow tip.
        If the middle_points_list is empty, a straight line (with an inserted interim midpoint
        if needed) is used.
      - If round is True, any inflexion (elbow) is smoothed using a quadratic Bézier curve
        approximation with a fixed number of intermediate points.
      - An arrowhead computed from the last shaft segment. The arrowhead is drawn as a continuous
        stroke that goes from the arrow tip to the left corner, back to the arrow tip, then to the right corner.

    Parameters:
      base_x, base_y      : Absolute coordinates offset.
      start_rel           : Starting point relative to the base, e.g. [x, y].
      end_rel             : Ending point (arrow tip) relative to the base.
      arrowhead_length    : Distance from the arrow tip back along the last shaft segment where the arrowhead begins.
      arrowhead_width     : Overall width of the arrowhead.
      middle_points_list  : List of relative points (each as [x, y]) for elbows.
                            If empty, the arrow is straight.
      round               : Boolean; if True, smooth out the sharp elbows with a curved transition.

    Returns:
      A list containing one stroke (a list of points, each as [x, y]). The stroke consists of the arrow shaft
      (with optional smoothing) and the arrowhead. The arrowhead is drawn by continuing from the shaft:
         ... shaft (ending at the arrow tip) -> left corner -> arrow tip -> right corner.
    """

    # Helper function to smooth a polyline at its interior vertices using quadratic Bézier curves.
    def smooth_polyline(points, smooth_steps=5, t=0.3):
        if len(points) < 3:
            return points
        new_points = [points[0]]
        for i in range(1, len(points) - 1):
            p_prev = points[i - 1]
            p_curr = points[i]
            p_next = points[i + 1]
            # Compute offset points along the incoming and outgoing segments
            p_in = [p_curr[0] - t * (p_curr[0] - p_prev[0]),
                    p_curr[1] - t * (p_curr[1] - p_prev[1])]
            p_out = [p_curr[0] + t * (p_next[0] - p_curr[0]),
                     p_curr[1] + t * (p_next[1] - p_curr[1])]
            new_points.append(p_in)
            # Generate intermediate points along the quadratic Bézier curve from p_in to p_out with control point p_curr.
            for s in range(1, smooth_steps):
                u = s / smooth_steps
                bx = (1 - u) ** 2 * p_in[0] + 2 * (1 - u) * u * p_curr[0] + u ** 2 * p_out[0]
                by = (1 - u) ** 2 * p_in[1] + 2 * (1 - u) * u * p_curr[1] + u ** 2 * p_out[1]
                new_points.append([bx, by])
            new_points.append(p_out)
        new_points.append(points[-1])
        return new_points

    # Build the arrow shaft polyline (absolute coordinates)
    shaft_points = []
    # Start point
    shaft_points.append([base_x + start_rel[0], base_y + start_rel[1]])
    # Add any middle (elbow) points, if provided
    for pt in middle_points_list:
        shaft_points.append([base_x + pt[0], base_y + pt[1]])
    # End point (arrow tip)
    shaft_points.append([base_x + end_rel[0], base_y + end_rel[1]])

    # Ensure the shaft has at least 3 points. If not, insert a midpoint.
    if len(shaft_points) < 3:
        mid = [(shaft_points[0][0] + shaft_points[1][0]) / 2,
               (shaft_points[0][1] + shaft_points[1][1]) / 2]
        shaft_points.insert(1, mid)

    # If round is True, smooth the corners (elbows) of the polyline.
    if round:
        shaft_points = smooth_polyline(shaft_points, smooth_steps=10, t=0.3)

    # Compute arrowhead based on the last segment of the shaft.
    arrow_tip = shaft_points[-1]
    second_last = shaft_points[-2]
    dx = arrow_tip[0] - second_last[0]
    dy = arrow_tip[1] - second_last[1]
    seg_length = math.hypot(dx, dy)
    if seg_length == 0:
        # Avoid division by zero by defaulting to a horizontal arrow.
        ux, uy = 1, 0
    else:
        ux, uy = dx / seg_length, dy / seg_length
    # Perpendicular vector (rotated 90° counterclockwise)
    perp = [-uy, ux]
    # Base of the arrowhead (along the last segment)
    base_arrow = [arrow_tip[0] - arrowhead_length * ux, arrow_tip[1] - arrowhead_length * uy]
    # Left and right corners for the arrowhead base
    left = [base_arrow[0] + (arrowhead_width / 2) * perp[0],
            base_arrow[1] + (arrowhead_width / 2) * perp[1]]
    right = [base_arrow[0] - (arrowhead_width / 2) * perp[0],
             base_arrow[1] - (arrowhead_width / 2) * perp[1]]

    # Combine the arrow shaft with the arrowhead.
    # The final stroke is: shaft (ending at arrow tip) -> left corner -> arrow tip -> right corner.
    combined = shaft_points + [left, arrow_tip, right]

    return [combined]


def get_diamond_points(base_x, base_y, width, height, angle=0):
    """
    Compute a continuous pen stroke for drawing a diamond with interim points,
    factoring in a rotation by the given angle (in radians) about the diamond's center.

    Without rotation, the diamond's vertices (in absolute coordinates) are:
      - Top:    (base_x + width/2, base_y)
      - Right:  (base_x + width, base_y + height/2)
      - Bottom: (base_x + width/2, base_y + height)
      - Left:   (base_x, base_y + height/2)

    The diamond is rotated about the center of the bounding rectangle.
    Each edge of the diamond is drawn with an interim (mid) point so that each stroke has at least three points.
    The continuous stroke is:
      Top -> (midpoint between Top and Right) -> Right ->
      (midpoint between Right and Bottom) -> Bottom ->
      (midpoint between Bottom and Left) -> Left ->
      (midpoint between Left and Top) -> back to Top.

    Parameters:
      base_x, base_y: Top-left corner of the bounding rectangle.
      width, height: Dimensions of the bounding rectangle.
      angle: Rotation angle in radians (counterclockwise, default 0).

    Returns:
      A list containing one stroke (a list of points). Each point is a list [x, y].
    """
    # Compute the center of the bounding rectangle
    center_x = base_x + width / 2
    center_y = base_y + height / 2

    # Define diamond vertices in local coordinates relative to the center:
    # (for a diamond, the vertices are the midpoints of the rectangle's edges)
    top_local = [0, -height / 2]
    right_local = [width / 2, 0]
    bottom_local = [0, height / 2]
    left_local = [-width / 2, 0]

    # Define interim midpoints in local coordinates
    mid_top_right_local = [(top_local[0] + right_local[0]) / 2, (top_local[1] + right_local[1]) / 2]
    mid_right_bottom_local = [(right_local[0] + bottom_local[0]) / 2, (right_local[1] + bottom_local[1]) / 2]
    mid_bottom_left_local = [(bottom_local[0] + left_local[0]) / 2, (bottom_local[1] + left_local[1]) / 2]
    mid_left_top_local = [(left_local[0] + top_local[0]) / 2, (left_local[1] + top_local[1]) / 2]

    # Helper function to rotate a point (x, y) by a given angle (in radians) about the origin,
    # then translate by the center.
    def transform(point):
        x, y = point
        x_rot = x * math.cos(angle) - y * math.sin(angle)
        y_rot = x * math.sin(angle) + y * math.cos(angle)
        return [x_rot + center_x, y_rot + center_y]

    # Transform all local points to absolute coordinates
    top = transform(top_local)
    right = transform(right_local)
    bottom = transform(bottom_local)
    left = transform(left_local)
    mid_top_right = transform(mid_top_right_local)
    mid_right_bottom = transform(mid_right_bottom_local)
    mid_bottom_left = transform(mid_bottom_left_local)
    mid_left_top = transform(mid_left_top_local)

    # Build the continuous stroke (closed shape)
    stroke = [
        top,
        mid_top_right,
        right,
        mid_right_bottom,
        bottom,
        mid_bottom_left,
        left,
        mid_left_top,
        top  # closing the shape
    ]

    return [stroke]


def get_rectangle_points0(base_x, base_y, width, height, angle=0):
    """
    Compute a continuous pen stroke for drawing a rectangle with interim points,
    factoring in a rotation by the given angle (in radians) about the rectangle's center.

    The rectangle is defined by its top-left corner (base_x, base_y), width, and height.
    The stroke starts at the top-left corner, then proceeds along each edge with an interim midpoint,
    and finally closes the rectangle by returning to the starting point.

    Parameters:
      base_x, base_y : The top-left corner of the bounding rectangle.
      width, height  : The dimensions of the rectangle.
      angle          : Rotation angle in radians (counterclockwise, default 0).

    Returns:
      A list containing one stroke (a list of points). Each point is represented as a list [x, y].
    """
    # Calculate the center of the rectangle
    center_x = base_x + width / 2
    center_y = base_y + height / 2

    # Define the rectangle corners in local coordinates relative to the center
    top_left_local = [-width / 2, -height / 2]
    top_right_local = [width / 2, -height / 2]
    bottom_right_local = [width / 2, height / 2]
    bottom_left_local = [-width / 2, height / 2]

    # Define midpoints for each edge in local coordinates
    mid_top_local = [0, -height / 2]
    mid_right_local = [width / 2, 0]
    mid_bottom_local = [0, height / 2]
    mid_left_local = [-width / 2, 0]

    # Helper function to rotate a point (x, y) by angle around the origin (0, 0)
    def rotate_point(point, theta):
        x, y = point
        x_rot = x * math.cos(theta) - y * math.sin(theta)
        y_rot = x * math.sin(theta) + y * math.cos(theta)
        return [x_rot, y_rot]

    # Rotate all local points by the given angle
    top_left_rot = rotate_point(top_left_local, angle)
    top_right_rot = rotate_point(top_right_local, angle)
    bottom_right_rot = rotate_point(bottom_right_local, angle)
    bottom_left_rot = rotate_point(bottom_left_local, angle)
    mid_top_rot = rotate_point(mid_top_local, angle)
    mid_right_rot = rotate_point(mid_right_local, angle)
    mid_bottom_rot = rotate_point(mid_bottom_local, angle)
    mid_left_rot = rotate_point(mid_left_local, angle)

    # Translate rotated points to absolute coordinates using the center
    def translate(point):
        return [point[0] + center_x, point[1] + center_y]

    top_left = translate(top_left_rot)
    top_right = translate(top_right_rot)
    bottom_right = translate(bottom_right_rot)
    bottom_left = translate(bottom_left_rot)
    mid_top = translate(mid_top_rot)
    mid_right = translate(mid_right_rot)
    mid_bottom = translate(mid_bottom_rot)
    mid_left = translate(mid_left_rot)

    # Combine points into one continuous stroke.
    # The stroke starts at the top-left, then goes:
    # top-left -> mid_top -> top-right -> mid_right ->
    # bottom-right -> mid_bottom -> bottom-left -> mid_left -> back to top-left.
    stroke = [
        top_left, mid_top, top_right, mid_right,
        bottom_right, mid_bottom, bottom_left, mid_left, top_left
    ]

    return [stroke]


def get_rectangle_points(base_x, base_y, width, height, angle=0, round=False):
    """
    Compute a continuous pen stroke for drawing a rectangle with interim points,
    factoring in a rotation by the given angle (in radians) about the rectangle's center.

    If round is True, the sharp corners of the rectangle are smoothed using a quadratic
    Bézier curve approximation.

    Parameters:
      base_x, base_y : The top-left corner of the bounding rectangle.
      width, height  : The dimensions of the rectangle.
      angle          : Rotation angle in radians (counterclockwise, default 0).
      round          : Boolean; if True, round the corners.

    Returns:
      A list containing one stroke (a list of points). Each point is represented as [x, y].
    """
    import math  # Make sure to import math

    # Calculate the center of the rectangle.
    center_x = base_x + width / 2
    center_y = base_y + height / 2

    # Define the rectangle corners in local coordinates relative to the center.
    top_left_local = [-width / 2, -height / 2]
    top_right_local = [width / 2, -height / 2]
    bottom_right_local = [width / 2,  height / 2]
    bottom_left_local = [-width / 2,  height / 2]

    # Define midpoints for each edge in local coordinates.
    mid_top_local = [0, -height / 2]
    mid_right_local = [width / 2, 0]
    mid_bottom_local = [0, height / 2]
    mid_left_local = [-width / 2, 0]

    # Helper: rotate a point by theta (radians) about the origin.
    def rotate_point(point, theta):
        x, y = point
        return [x * math.cos(theta) - y * math.sin(theta),
                x * math.sin(theta) + y * math.cos(theta)]

    # Rotate all local points.
    top_left_rot = rotate_point(top_left_local, angle)
    top_right_rot = rotate_point(top_right_local, angle)
    bottom_right_rot = rotate_point(bottom_right_local, angle)
    bottom_left_rot = rotate_point(bottom_left_local, angle)
    mid_top_rot = rotate_point(mid_top_local, angle)
    mid_right_rot = rotate_point(mid_right_local, angle)
    mid_bottom_rot = rotate_point(mid_bottom_local, angle)
    mid_left_rot = rotate_point(mid_left_local, angle)

    # Translate rotated points to absolute coordinates.
    def translate(point):
        return [point[0] + center_x, point[1] + center_y]

    top_left = translate(top_left_rot)
    top_right = translate(top_right_rot)
    bottom_right = translate(bottom_right_rot)
    bottom_left = translate(bottom_left_rot)
    mid_top = translate(mid_top_rot)
    mid_right = translate(mid_right_rot)
    mid_bottom = translate(mid_bottom_rot)
    mid_left = translate(mid_left_rot)

    # Build the stroke.
    # For non-rounded version, we want a closed polyline (so we repeat the starting point).
    # For the rounded version we start with an open polyline and let the smoothing function treat it as closed.
    stroke = [
        top_left, mid_top, top_right, mid_right,
        bottom_right, mid_bottom, bottom_left, mid_left
    ]
    if not round:
        stroke.append(top_left)  # close the polyline

    # Helper: smooth a closed polyline using quadratic Bézier curves.
    def smooth_closed_polyline(points, smooth_steps=10, t=0.3):
        # Assume 'points' is an open list but treat it as closed (cyclic).
        n = len(points)
        if n < 3:
            return points
        new_points = []
        for i in range(n):
            # Use cyclic indexing for previous and next.
            p_prev = points[i - 1]  # works for i=0 because of negative indexing
            p_curr = points[i]
            p_next = points[(i + 1) % n]
            # Compute offset points along incoming and outgoing segments.
            p_in = [p_curr[0] - t * (p_curr[0] - p_prev[0]),
                    p_curr[1] - t * (p_curr[1] - p_prev[1])]
            p_out = [p_curr[0] + t * (p_next[0] - p_curr[0]),
                     p_curr[1] + t * (p_next[1] - p_curr[1])]
            new_points.append(p_in)
            # Generate intermediate points along quadratic Bézier curve.
            for s in range(1, smooth_steps):
                u = s / smooth_steps
                bx = (1 - u)**2 * p_in[0] + 2 * (1 - u) * u * p_curr[0] + u**2 * p_out[0]
                by = (1 - u)**2 * p_in[1] + 2 * (1 - u) * u * p_curr[1] + u**2 * p_out[1]
                new_points.append([bx, by])
            new_points.append(p_out)
        # Do NOT add a duplicate starting point at the end.
        return new_points

    if round:
        stroke = smooth_closed_polyline(stroke, smooth_steps=10, t=0.3)
        # Close the shape by appending the first point of the rounded top-left corner.
        stroke.append(stroke[0])

    return [stroke]


def get_ellipse_points(base_x, base_y, width, height, angle=0, steps=36):
    """
    Compute a continuous pen stroke for drawing an ellipse approximated by a series of points.
    The ellipse fits within the rectangle defined by (base_x, base_y, width, height) and is rotated
    by 'angle' degrees (counterclockwise) about the center of the rectangle.

    Parameters:
      base_x, base_y : The top-left corner of the bounding rectangle.
      width, height  : The dimensions of the bounding rectangle.
      angle          : Rotation angle in degrees (default 0).
      steps          : Number of points to approximate the ellipse (default 36).

    Returns:
      A list containing one stroke (a list of points). Each point is a list [x, y].
      The stroke is closed by appending the first point at the end.
    """
    # Calculate the center of the ellipse
    center_x = base_x + width / 2
    center_y = base_y + height / 2

    # Radii in x and y directions
    rx = width / 2
    ry = height / 2

    # The angle is in radians
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    stroke = []
    for i in range(steps):
        theta = 2 * math.pi * i / steps
        # Compute point on the ellipse in local coordinates (centered at 0,0)
        x_local = rx * math.cos(theta)
        y_local = ry * math.sin(theta)
        # Rotate the local coordinates by the given angle
        x_rot = x_local * cos_a - y_local * sin_a
        y_rot = x_local * sin_a + y_local * cos_a
        # Translate to absolute coordinates using the center
        x_abs = center_x + x_rot
        y_abs = center_y + y_rot
        stroke.append([x_abs, y_abs])

    # Close the ellipse by appending the first point at the end
    stroke.append(stroke[0])
    return [stroke]


def unique_random(bound=pysn.XC_VERSION_MAX):
    while True:
        a_pick = random.randint(1, bound)
        if a_pick not in unique_list:
            unique_list.append(a_pick)
            break
        if len(unique_list) >= bound:
            break
    return a_pick


def unique_update():

    a_time = int(time.time() * 1000)
    if a_time in update_list:
        a_time += 1
    update_list.append(a_time)

    return a_time


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
        series, num_pages, resolutions=pysn.XC_RESOLUTIONS, horizontal_pages=pysn.XC_HORIZONTAL_PAGES, spacing=pysn.XC_PAGE_SPACING,
        initial_x=pysn.XC_INITIAL_X, initial_y=pysn.XC_INITIAL_Y, ocr_layer=False, screen_ratio=pysn.XC_SCREEN_RATIO):
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
    # Initialize the excalidraw variable
    excalidraw_file = {}
    elements = []
    page_mapping = {}
    offset = 0

    # Check if series contains a filename or a series. If a series, it's a new template
    new_template = series in resolutions

    if new_template:
        # Build the overall Excalidraw file structure.
        excalidraw_file = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "appState": {
                "gridSize": 20,
                "gridStep": 5,
                "gridModeEnabled": False,
                "viewBackgroundColor": "#ffffff"
            },
            "files": {}}
    else:
        # Load the Excalidraw file.
        with open(series, 'r', encoding='utf-8', errors="replace") as f:
            excalidraw_file = json.load(f)
        elements = excalidraw_file.get("elements", [])
        page_elements = [elem for elem in elements if elem.get("type") == "frame" and elem.get("name")[:5] == "Page "]

        offset = len(page_elements)

        if offset > 0:
            page_1 = page_elements[0]
            series = page_1.get('pysn', 'N5')
            screen_ratio = page_1.get('screen_ratio', screen_ratio)

    frame_width, frame_height = resolutions[series]
    frame_width = int(frame_width/screen_ratio)
    frame_height = int(frame_height/screen_ratio)

    for i in range(offset, num_pages + offset):
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
            "version": random.randint(1, 10000),
            "versionNonce": random.randint(1, 1000000000),
            "isDeleted": False,
            "boundElements": [],
            "updated": unique_update(),
            "link": None,
            "locked": True,
            "name": f"Page {i+1}",
            "pysn": series,
            "horizontal": pysn.XC_HORIZONTAL_PAGES,
            "screen_ratio": screen_ratio

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
                "version": random.randint(1, 10000),
                "versionNonce": random.randint(1, 1000000000),
                "isDeleted": False,
                "boundElements": [],
                "updated": unique_update(),
                "link": None,
                "locked": False,
                "name": "",
                "pysn": series,
                "horizontal": pysn.XC_HORIZONTAL_PAGES,
                "screen_ratio": screen_ratio
                }
            elements.append(ocr_frame)
        page_mapping[i + 1] = frame

    excalidraw_file["elements"] = elements
    return excalidraw_file, page_mapping, new_template


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
        screen_ratio=pysn.XC_SCREEN_RATIO):
    """
    Add a freedraw element (hand-drawn stroke) to a specific page (frame) in the Excalidraw file from
    a pen stroke dict.

    Args:
        page_number (int): The page number (as set in the frame's "name", e.g., "Page 1").
        excalidraw_file (dict): The Excalidraw file structure (as returned by create_pages).

    """
    try:

        max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = pysn.series_bounds(series)
        # retrieve the color
        try:
            color = pysn.XC_COLORS[pen_stroke['color']]
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

        normalized_points = [
            pysn.topright_to_topleft(
                x, max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y) for x in stroke_vector_points]

        # Find the frame corresponding to the given page number.
        vector_points = [(round((x[0]-min_c_x)/screen_ratio, 6), round((x[1]-min_c_y)/screen_ratio), 6) for x in normalized_points]

        # Set the drawing's position relative to the frame.
        base_width = target_frame.get("width", 0)
        base_height = target_frame.get("height", 0)

        # Frame Positionning
        denominator_x = pysn.XC_RESOLUTIONS[series][0]
        denominator_y = pysn.XC_RESOLUTIONS[series][1]

        if pysn.XC_HORIZONTAL_PAGES:
            freedraw_x = round((page_number - 1)*(pysn.XC_PAGE_SPACING + base_width) + min_c_x + base_width * min_c_x / denominator_x, 6)
            freedraw_y = round(min_c_y+base_height*min_c_y/denominator_y, 6)
        else:
            freedraw_x = round(min_c_x+base_width*min_c_x/denominator_x, 6)
            freedraw_y = round((page_number-1)*(pysn.XC_PAGE_SPACING+base_height)+min_c_y+base_height*min_c_y/denominator_y, 6)

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

        if width <= 0 or height <= 0:
            return

        vector_points = [(x[0]-min_c_x, x[1]-min_c_y) for x in vector_points]

        # Remove adjacent duplicates using list comprehension and enumerate
        vector_points = [pt for i, pt in enumerate(vector_points) if i == 0 or pt != vector_points[i-1]]

        # Create the freedraw element.
        freedraw_element = {
            "id": generate_excalidraw_id(use_uuid=False),
            "type": "freedraw",
            "x": freedraw_x,
            "y": freedraw_y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": pen_stroke_width*max(1, 3 - screen_ratio),
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": target_frame["id"],
            "index": f"a{unique_random()}",  # Arbitrary index for ordering.
            "roundness": None,
            "seed": unique_random(bound=pysn.XC_MAX_UINT32),
            "version": unique_random(),
            "versionNonce": unique_random(bound=pysn.XC_MAX_UINT32),
            "isDeleted": False,
            "boundElements": None,
            "updated": unique_update(),
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


def add_text2canvas(target_frame, page_number, excalidraw_file, word_rect_std, word_text, screen_ratio=pysn.XC_SCREEN_RATIO):
    """
    Add ocr to an ocr frame pertaining to a page.

    """

    # Find the frame corresponding to the given page number.

    word_rect_std = [int(x/screen_ratio) for x in word_rect_std]

    min_c_x, min_c_y, width, height = word_rect_std

    # Set the drawing's position relative to the frame.
    base_width = target_frame.get("width", 0)
    base_height = target_frame.get("height", 0)

    if pysn.XC_HORIZONTAL_PAGES:
        text_x = round((int(page_number) - 1)*(pysn.XC_PAGE_SPACING + base_width) + min_c_x)
        text_y = round(min_c_y)
    else:
        text_x = round(min_c_x)
        text_y = round((int(page_number)-1)*(pysn.XC_PAGE_SPACING+base_height) + min_c_y)

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
        "index": f"a{unique_random()}",  # Arbitrary index for ordering.
        "roundness": None,
        "seed": random.randint(0, 1000000000),
        "version": random.randint(1, 1000),
        "versionNonce": random.randint(0, 1000000000),
        "isDeleted": False,
        "boundElements": None,
        "updated": unique_update(),
        "link": None,
        "locked": False,
        "text": word_text,
        "fontSize": 32*max(1, 3-screen_ratio),
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


def note2ex(note_fn, blank_pages=0, screen_ratio=pysn.XC_SCREEN_RATIO):
    """
    Input:
        - filename of a Supernote notebook
        - series of the Supernote device that created the notebook
          ('N5' for Manta, 'N6' for Nomad)
    Output:
        - An Excalidraw file
    """
    try:
        if blank_pages != 0:
            try:
                exca_file, _, new_template = create_pages(note_fn, blank_pages, ocr_layer=False, screen_ratio=screen_ratio)
                # if the filename is a series, we know it's a blank template
                if new_template:
                    # Get the current working directory
                    current_directory = os.getcwd()
                    # Define the path for the new directory
                    blank_directory = os.path.join(current_directory, 'blank')
                    # Check if the directory already exists
                    if not os.path.exists(blank_directory):
                        # Create the directory
                        os.mkdir(blank_directory)
                    output_fn = os.path.join(blank_directory, f'{note_fn}_{blank_pages}_pager.excalidraw')
                    print(f'  > Generated file: {output_fn}')
                else:
                    output_fn = note_fn
                    print(f'  > Updated file: {output_fn}')
                pysn.save_json(output_fn, exca_file)

            except Exception as e:
                print(f'*** Creating pages: {e}')
            exit(1)

        print(f'Processing file: {note_fn}')

        pen_strokes_dict, _, meta_data, series = pysn.get_pen_strokes_dict(note_fn)

        _, ocr_pages_dict = pysn.titles_and_text_from_notes(meta_data, note_fn)

        ocr_file = ocr_pages_dict != {}

        page_nb = len(pen_strokes_dict.keys())

        exca_file, _, _ = create_pages(series, page_nb, ocr_layer=ocr_file, screen_ratio=screen_ratio)

        # Parse the dictionary of totalpath objects

        for apage, avalue in pen_strokes_dict.items():

            target_frame, target_frame_ocr = page_number2frame(exca_file, apage, ocr=ocr_file)

            page_number = int(apage)

            a_list_strokes = avalue['strokes']

            for a_stroke in a_list_strokes:
                add_ps2canvas(
                    a_stroke, target_frame, page_number, exca_file, series, screen_ratio=screen_ratio)

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
                                        word_rect_std = [x*pysn.XC_MYSCRIPT_RATIO for x in word_rect]
                                        word_text = a_word['label']
                                        if word_text != '':
                                            add_text2canvas(target_frame_ocr, page_number, exca_file, word_rect_std, word_text, screen_ratio=screen_ratio)

        output_fn = f'{note_fn[:-5]}.excalidraw'

        pysn.save_json(output_fn, exca_file)
        print(f'Generated file: {output_fn}')
    except Exception as e:
        print(f'*** note2ex: {e}')
        print(ocr_file)


def extract_penstrokes_by_page(excalidraw_filename, screen_ratio, page_elements={}):
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

    with open(excalidraw_filename, 'r', encoding='utf-8', errors="replace") as f:
        excalidraw_file = json.load(f)

    # Build a mapping from frame id to frame properties (including x, y positions).

    initialization = True
    frame_mapping = {}

    for elem in excalidraw_file.get("elements", []):
        if elem.get("type") == "frame":

            if initialization:
                # Check if this is valid SNEX/PySN scene
                series = elem.get("pysn", None)
                screen_ratio_ = elem.get("screen_ratio", pysn.XC_SCREEN_RATIO)
                if screen_ratio_ != screen_ratio:
                    print(f'   > WARNING: Using file screen ratio ({screen_ratio_}) instead of your default ({screen_ratio}) <')
                    screen_ratio = screen_ratio_

                if series is None:
                    print()
                    print(f'*** No Supernote series found in {excalidraw_filename}. Was it created by SNEX?')
                    print('    > Exiting because Supernote requires specific implementation of pages through frames <')
                    exit(1)
                pysn.XC_HORIZONTAL_PAGES = elem.get("horizontal", pysn.XC_HORIZONTAL_PAGES)
                if pysn.XC_HORIZONTAL_PAGES:
                    print('   > Loading a scene with horizontal pages')
                else:
                    print('   > Loading a scene with vertical pages')
                print()
                max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = pysn.series_bounds(series)
                initialization = False

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

    # Process freedraw elements.
    elements = excalidraw_file.get("elements", [])
    crowded_canvas = len(elements) > pysn.XC_CROWD_THRESHOLD

    if crowded_canvas:
        print(f'   - Canvas with more than {pysn.XC_CROWD_THRESHOLD} elements -> I will use thinnest pen stroke')

    # Parsing all elements, skipping what is not freedraw or line
    # TODO: We commingle 'line' and freedraw because vector points of images often use 'line' object. To be revisited
    for elem in elements:
        element_type = elem.get("type")
        if element_type not in ["freedraw", "line"]:
            continue

        frame_id = elem.get("frameId")

        if crowded_canvas:
            stroke_width = 0.3
        else:
            stroke_width = elem.get("strokeWidth", 0)

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

        # First, compute the freedraw element's position relative to the frame.
        relative_x = elem["x"] - frame_x
        relative_y = elem["y"] - frame_y

        # Invert the positioning transformation.
        if pysn.XC_HORIZONTAL_PAGES:
            recovered_min_c_x = relative_x - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_width)
            recovered_min_c_y = relative_y
        else:
            recovered_min_c_x = relative_x
            recovered_min_c_y = relative_y - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_height)

        # Multiply by screen_ratio to recover original e-ink coordinates.
        orig_x = recovered_min_c_x * screen_ratio
        orig_y = recovered_min_c_y * screen_ratio
        orig_width = elem.get("width", 0) * screen_ratio
        orig_height = elem.get("height", 0) * screen_ratio
        recovered_points = []

        for pt in elem.get("points", []):
            temp_x = pt[0] + relative_x
            temp_y = pt[1] + relative_y

            if pysn.XC_ROUND_COORDINATES:
                normalized_x = round(temp_x * screen_ratio)
                normalized_y = round(temp_y * screen_ratio)
            else:
                normalized_x = temp_x * screen_ratio
                normalized_y = temp_y * screen_ratio

            normalized_point = [normalized_x, normalized_y]
            recovered_points.append(normalized_point)

        # Remove adjacent duplicates using list comprehension and enumerate
        recovered_points = [pt for i, pt in enumerate(recovered_points) if i == 0 or pt != recovered_points[i-1]]

        # Build the dictionary for this text element.
        freedraw_dict = {
            "type": "freedraw",
            "x": orig_x,
            "y": orig_y,
            "width": orig_width,
            "height": orig_height,
            "strokeWidth": stroke_width,
            "color": elem.get("strokeColor"),
            "points": recovered_points
        }

        page_elements.setdefault(page_number, []).append(freedraw_dict)

    return page_elements, series, screen_ratio, frame_mapping


def extract_text_by_page(excalidraw_filename, page_elements={}, screen_ratio=pysn.XC_SCREEN_RATIO):
    """
    Load an Excalidraw file from disk and extract text elements (OCR) for each page.
    Each page key maps to a list of dictionaries, each having keys:
      x, y, width, height, color, text.
    These values represent the original OCR rectangle (as defined in word_rect_std) in the e-ink coordinate space.

    Args:
        excalidraw_filename (str): Path to the Excalidraw JSON file on disk.
        series: Identifier used for resolution lookup (RESOLUTIONS[series] must be defined).
        screen_ratio (float): The screen ratio used during the forward transformation.

    Returns:
        dict: Keys are page numbers (extracted from frame names like "Page 1"), and values are lists
              of dictionaries, each with keys: x, y, width, height, color, text.
    """
    # Load the Excalidraw file.

    with open(excalidraw_filename, 'r', encoding='utf-8', errors="replace") as f:
        excalidraw_file = json.load(f)

    # Build a mapping from frame id to its properties.
    # We record page_number (from the frame's "name"), width, height, and absolute x, y.

    frame_mapping = {}
    fonts_dict = {}
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
                    "x": elem.get("x", 0),  # Frame's absolute x
                    "y": elem.get("y", 0),  # Frame's absolute y
                }

    # Process text elements.
    for elem in excalidraw_file.get("elements", []):
        if elem.get("type") != "text":
            continue

        frame_id = elem.get("frameId")
        if frame_id not in frame_mapping:
            # Skip text elements not attached to a recognized frame.
            continue

        # Retrieve frame parameters.
        frame_props = frame_mapping[frame_id]
        page_number = frame_props["page_number"]
        base_width = frame_props["width"]
        base_height = frame_props["height"]
        # frame_x = frame_props["x"]
        # frame_y = frame_props["y"]

        # Compute the text element's position relative to its frame.
        # This ensures that if the frame was moved on the canvas, the recovered rectangle stays the same.

        # relative_x = elem["x"] - frame_x
        # relative_y = elem["y"] - frame_y

        relative_x = elem["x"]
        relative_y = elem["y"]

        # Invert the positioning transformation.
        if pysn.XC_HORIZONTAL_PAGES:
            # Forward: text_x = (page_number - 1)*(PAGE_SPACING + base_width) + min_c_x
            # Inverse: min_c_x = relative_x - (page_number - 1)*(PAGE_SPACING + base_width)
            recovered_min_c_x = relative_x - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_width)
            recovered_min_c_y = relative_y  # Because forward used: text_y = min_c_y
        else:
            # Forward: text_y = (page_number - 1)*(PAGE_SPACING + base_height) + min_c_y
            # Inverse: min_c_y = relative_y - (page_number - 1)*(PAGE_SPACING + base_height)
            recovered_min_c_x = relative_x  # Because forward used: text_x = min_c_x
            # recovered_min_c_y = relative_y - (page_number - 1) * (PAGE_SPACING + base_height)

            recovered_min_c_y = relative_y - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_height)

        # Multiply by screen_ratio to recover original e-ink coordinates.
        orig_x = recovered_min_c_x * screen_ratio
        orig_y = recovered_min_c_y * screen_ratio
        orig_width = elem.get("width", 0) * screen_ratio
        orig_height = elem.get("height", 0) * screen_ratio
        orig_text = elem.get("text").strip()
        orig_weight = round(elem.get("fontSize")*pysn.FONT_RATIOS[1])

        # Build the dictionary for this text element.
        font_family = elem.get("fontFamily")
        mapping_dict = globals()['pysn.XC_FONT_FAMILY']

        if str(font_family) in mapping_dict:
            mapped_font_family = mapping_dict[str(font_family)].split('/')[1]

            a_weight_key = f'{mapped_font_family}@{orig_weight*screen_ratio}'
            try:
                if a_weight_key in fonts_dict:
                    fonts_weight_adjustment = fonts_dict[a_weight_key]
                else:
                    fonts_adj_json_fn = os.path.join(pysn.CONFIG, f'{mapped_font_family}.json')
                    fonts_adjustment = pysn.read_json(fonts_adj_json_fn)
                    fonts_weight_adjustment_ = pysn.list_to_tree(fonts_adjustment['weight'])[(orig_weight-1.5)*screen_ratio]
                    interval_ = next(iter(fonts_weight_adjustment_))
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES[interval_.data]
                    fonts_dict[a_weight_key] = fonts_weight_adjustment
                print(f'     - {a_weight_key}: {orig_text[:15]} ...')
            except Exception as e:
                print(f'    **- Loading weight json for {a_weight_key}: {e}')
                if orig_weight <= 25:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["0.2"]
                elif orig_weight <= 40:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["0.3"]
                elif orig_weight <= 56:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["0.4"]
                elif orig_weight <= 70:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["0.5"]
                elif orig_weight <= 100:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["0.8"]
                elif orig_weight <= 120:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["1"]
                else:
                    fonts_weight_adjustment = pysn.NEEDLE_POINT_SIZES["2"]

        text_dict = {
            "type": "text",
            "x": orig_x,
            "y": orig_y,
            "width": orig_width,
            "height": orig_height,
            "color": elem.get("strokeColor"),
            "text": orig_text,
            "fontSize": fonts_weight_adjustment,
            "originalFontSize": orig_weight,
            "font_ratio": pysn.FONT_RATIOS[1],
            "fontFamily": mapped_font_family
        }

        page_elements.setdefault(page_number, []).append(text_dict)

    return page_elements


def extract_shapes_by_page(excalidraw_filename, page_elements={}, screen_ratio=pysn.XC_SCREEN_RATIO):
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

    with open(excalidraw_filename, 'r', encoding='utf-8', errors="replace") as f:
        excalidraw_file = json.load(f)

    # Build a mapping from frame id to frame properties (including x, y positions).
    frame_mapping = {}
    series = excalidraw_file.get("pysn", "N5")
    max_horizontal_pixels, max_vertical_pixels, adb_screen_max_x, adb_screen_max_y = pysn.series_bounds(series)
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

    # Process freedraw elements.
    for elem in excalidraw_file.get("elements", []):

        element_type = elem.get("type")

        if element_type not in ["arrow", "rectangle", "ellipse", "diamond"]:
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

        # First, compute the freedraw element's position relative to the frame.
        if element_type == "arrow":
            relative_x = elem["x"] - frame_x
            relative_y = elem["y"] - frame_y
        else:
            relative_x = elem["x"]
            relative_y = elem["y"]

        # Invert the positioning transformation.
        if pysn.XC_HORIZONTAL_PAGES:
            recovered_min_c_x = relative_x - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_width)
            recovered_min_c_y = relative_y
        else:
            recovered_min_c_x = relative_x
            recovered_min_c_y = relative_y - (page_number - 1) * (pysn.XC_PAGE_SPACING + base_height)

        # Multiply by screen_ratio to recover original e-ink coordinates.
        orig_x = recovered_min_c_x * screen_ratio
        orig_y = recovered_min_c_y * screen_ratio

        orig_width = elem.get("width", 0) * screen_ratio
        orig_height = elem.get("height", 0) * screen_ratio
        recovered_points = []

        round = False

        if element_type == "rectangle":
            round = elem.get("roundness", None) is not None

        elif element_type == "arrow":

            round = elem.get("elbowed", False)

            for pt in elem.get("points", []):
                temp_x = pt[0] + relative_x
                temp_y = pt[1] + relative_y

                normalized_x = temp_x * screen_ratio
                normalized_y = temp_y * screen_ratio
                normalized_point = [normalized_x, normalized_y]
                recovered_points.append(normalized_point)

        # Remove adjacent duplicates using list comprehension and enumerate
        recovered_points = [pt for i, pt in enumerate(recovered_points) if i == 0 or pt != recovered_points[i-1]]

        # Build the dictionary for this text element.
        freedraw_dict = {
            "type": element_type,
            "x": orig_x,
            "y": orig_y,
            "width": orig_width,
            "height": orig_height,
            "strokeWidth": elem.get("strokeWidth", 0),
            "angle": elem.get("angle", 0),
            "color": elem.get("strokeColor"),
            "points": recovered_points,
            "round": round,
            "strokeStyle": elem.get("strokeStyle", "")
        }

        page_elements.setdefault(page_number, []).append(freedraw_dict)

    return page_elements


def main():
    try:
        new_user_settings = None
        # Initialize conversion variable.
        do_conversion = True
        # Load the user_settings
        user_settings = pysn.load_user_settings()
        # Argument parser setup.
        epilog = ""
        parser = argparse.ArgumentParser(epilog=epilog, description="SNEX command line interpreter (CLI) for updating global variables.")

        if len(sys.argv) > 1:
            first_arg = sys.argv[1]
            if len(first_arg) > 1:
                if first_arg[:1] != '-':
                    parser.add_argument("filename", help="The file to process")

        for long_param, details in pysn.SETTINGS_DESC_DICT.items():  # Go through the list of global variables exposed to CLI

            a_variable_name = details['var']
            description = details['description']
            param_type = details['type']
            short_param = details['short']
            if 'choices' in details:
                choices = details['choices']
            else:
                choices = None

            a_variable_value = eval(f'pysn.{a_variable_name}')  # Retrieve the current variable value (the hard-coded value)

            if long_param in user_settings:
                a_variable_value = user_settings[long_param]  # Overwrite with user's saved values

            if param_type == bool:  # Add a '-no-' prefix option to set boolean values
                parser.add_argument(f'--{long_param}', action='store_true', default=a_variable_value, help=f'{description}. [%(default)s]')
                parser.add_argument(f'--no-{long_param}', action='store_true', default=None,  help='')

            elif long_param == 'input':

                parser.add_argument(
                    f'--{long_param}', nargs='+',  type=param_type, default=a_variable_value, metavar=f'{a_variable_name}',
                    help=f'{description}.  [{" ".join(a_variable_value)}]')

            else:
                if choices:
                    parser.add_argument(f'-{short_param}', f'--{long_param}', choices=choices, type=param_type, default=a_variable_value, help=f'{description}. [%(default)s]')
                else:
                    parser.add_argument(f'-{short_param}', f'--{long_param}', type=param_type, default=a_variable_value, help=f'{description}. [%(default)s]')

        # Add a reset option (back to hard-coded values)
        parser.add_argument('--reset', action='store_true', help='Reset to hard-coded values: the values that are affected to digest.py global variables (in UPPERCASE when using the help)')
        if 'pages' in user_settings:
            nbb_pages = user_settings['pages']
        else:
            nbb_pages = pysn.PAGES
        parser.add_argument('--blank', action='store_true', help=f"Creates a blank Excalidraw file with {nbb_pages} pages (use --page to change page #) \
                            for {pysn.NOTEBOOK_DEVICE} device type (use --device to change device type)")

        # Parse command line arguments
        try:
            args = parser.parse_args()
        except Exception as e:
            print(f'**-: {e}')

        # Convert args to a dictionary for easier handling
        args_dict = vars(args)

        if first_arg == '--reset':
            # The user has decided to revert settings to hard-coded values.
            new_user_settings = {"filename": "demo/xc2sn/version1.06.excalidraw"}
            pysn.save_user_settings(new_user_settings)
            print()
            print(f'SNEX Version {VERSION}')
            print('-----------------')
            print()
            print('  > Your settings have reverted to default (hard-coded) values')
            exit(1)
        else:
            # Now browse the args_back, handling boolean settings, if needed
            args_dict.pop('reset')
            args_dict.pop('blank')
            new_user_settings = dict(args_dict)
            for a_key, a_value in args_dict.items():
                if type(a_value) is bool:
                    a_key_list = a_key.split('_')
                    if 'no' == a_key_list[0]:
                        base_key = '_'.join(a_key_list[1:])
                        new_user_settings[base_key] = False
                        new_user_settings.pop(a_key)
                elif a_value is None:
                    new_user_settings.pop(a_key)

            # Now altering global variables

            for a_key, a_value in new_user_settings.items():

                a_global_var_name = f"pysn.{pysn.SETTINGS_DESC_DICT[a_key]['var']}"
                a_type = type(a_value)
                if a_type in [str, float, dict, int]:

                    if a_value != "":
                        globals()[a_global_var_name] = a_value

                else:
                    globals()[a_global_var_name] = a_value

        pysn.save_user_settings(new_user_settings)  # Saving the last set of settings in 'user_settings.json'

        print()
        print(f'SNEX Version {VERSION}')
        print('-----------------')
        print()
        filename = globals()['pysn.FILE_NAME']
        screen_ratio = globals()['pysn.XC_SCREEN_RATIO']

        if '--pages' in sys.argv or '-pages' in sys.argv:
            do_conversion = False
            print(f'  > Default # of pages for blank excalidraw files is now: {globals()['pysn.PAGES']} page(s)')

        if '--xc_screen_ratio' in sys.argv or '-xcsr' in sys.argv:
            do_conversion = False
            print(f'  > Default screen ratio is now: {globals()['pysn.XC_SCREEN_RATIO']}')

        if '--device' in sys.argv or '-device' in sys.argv:
            do_conversion = False
            print(f'  > Default device is now: {globals()['pysn.NOTEBOOK_DEVICE']}')

        if '--font_name' in sys.argv or '-font_name' in sys.argv:
            do_conversion = False
            print(f'  > Default font is now: {globals()['pysn.FONT_NAME']}')

        if '--xcns' in sys.argv or '-xcns' in sys.argv:
            do_conversion = False
            print(f'  > SVG number of sample points per segment is now: {globals()['pysn.XC_NUM_SAMPLES']}')

        if '--xcstr' in sys.argv or '-xcstr' in sys.argv:
            do_conversion = False
            print(f'  > Thick/thin ratio for SVG pen weight is now: {globals()['pysn.XC_SVG_THIN_RATIO']}')

        if '--blank' in sys.argv or '-blank' in sys.argv:
            do_conversion = False
            blank_pages = globals()['pysn.PAGES']
            series = globals()['pysn.NOTEBOOK_DEVICE']

            # Checking if filename was provided (ie. filename != versionfile)
            filename_provided = f'version{VERSION}' not in filename
            if filename_provided:
                print(f'  > Adding {blank_pages} blank page(s) to {filename}')
                note2ex(filename, blank_pages=int(blank_pages), screen_ratio=screen_ratio)
            else:
                print(f'  > Creating {blank_pages} blank page(s) for {series} series and screen ratio {screen_ratio}')
                note2ex(series, blank_pages=int(blank_pages), screen_ratio=screen_ratio)

        if do_conversion:

            # Get the file extension
            basename, file_extension = os.path.splitext(filename)

            if not os.path.exists(filename):
                print()
                print(f'*** File not found: {filename}')
                print('    > Please check the path. I am exiting ... <')

            elif file_extension.lower() == '.note':
                note2ex(filename, screen_ratio=screen_ratio)
            elif file_extension.lower() == '.excalidraw':
                adict, series, screen_ratio, frame_mapping = extract_penstrokes_by_page(filename, screen_ratio)
                adict = extract_text_by_page(filename, adict, screen_ratio=screen_ratio)
                adict = extract_shapes_by_page(filename, adict, screen_ratio=screen_ratio)
                adict, templates_dict = load_excalidraw_images(filename, frame_mapping, series, screen_ratio, adict)

                output_fn = f'{basename}.json'

                # Add advertisement pen-strokes (templates are not displaying for empty pages)
                if adict == {}:
                    adict = {
                        1: [
                            {'type': 'text',
                             'x': 1370,
                             'y': 120,
                             'width': 425,
                             'height': 45,
                             'color': '#1e1e1e',
                             'text': f'Generated by SNEX version {VERSION}',
                             'fontSize': 200,
                             'originalFontSize': 18,
                             'font_ratio': 0.55,
                             'fontFamily': 'barlow'}]}

                if pysn.DEBUG_MODE:
                    pysn.save_json(output_fn, adict)
                    print(f'Generated file: {output_fn}')

                full_dict_ps = {}
                titles_dict = {}
                images_dict = {}
                temp_dict = {}
                font_maily_dict = {}

                ascii_ps_list, _ = pysn.get_pen_stokes_list_from_table(series)
                font_maily_dict[pysn.FONT_NAME] = ascii_ps_list
                if len(adict.keys()) > 0:

                    print()
                    print('   > Generating pen strokes from elements...')
                    for a_page, list_elements in adict.items():
                        page_nb_str = str(a_page)
                        print(f'     - page {a_page}')

                        for an_element in list_elements:
                            paths_list = []

                            an_element_type = an_element["type"]

                            el_color = an_element['color']
                            if el_color in pysn.XC_COLORS_O:
                                a_color = pysn.XC_COLORS_O[el_color]
                            else:
                                a_color = None

                            if an_element_type == "text":
                                # Handling text element
                                a_text = an_element['text']
                                a_width = an_element['width']
                                a_height = an_element['height']
                                a_font_size = an_element['originalFontSize']
                                a_font_family = an_element['fontFamily']

                                new_weight = an_element['fontSize']

                                starting_point = [an_element['x'], an_element['y']]

                                if a_font_family in font_maily_dict:
                                    ascii_ps_list = font_maily_dict[a_font_family]
                                else:
                                    ascii_ps_list, _ = pysn.get_pen_stokes_list_from_table(series, font_name=a_font_family)
                                    font_maily_dict[a_font_family] = ascii_ps_list

                                full_dict_ps, _, images_dict = pysn.text_to_pen_strokes_nf(
                                    full_dict_ps, titles_dict, images_dict,
                                    a_text, ascii_ps_list, starting_point, a_page, temp_dict=temp_dict, series=series, new_color=a_color,
                                    width=a_width, height=a_height, new_weight=new_weight, new_font_size=a_font_size,
                                    font_family=a_font_family, screen_ratio=screen_ratio)

                            elif an_element_type in ["freedraw", "arrow", "rectangle", "ellipse", "diamond"]:
                                # Handling pen strokes and shapes. TODO: unmerge the 'line' object. For now it's merged with freedraw
                                pen_size = an_element["strokeWidth"]
                                if pen_size <= 0.5:
                                    pen_size = "0.1"
                                elif pen_size <= 1:
                                    pen_size = "0.3"
                                elif pen_size <= 2:
                                    pen_size = "0.5"
                                elif pen_size <= 3:
                                    pen_size = "1"
                                else:
                                    pen_size = "2"

                                if 'strokeStyle' in an_element:
                                    dashed = an_element['strokeStyle'] == 'dashed'
                                    dotted = an_element['strokeStyle'] == 'dotted'
                                else:
                                    dashed = False
                                    dotted = False

                                if an_element_type in ["freedraw", "line"]:
                                    paths_list.append(an_element['points'])

                                elif an_element_type == "arrow":
                                    arrow_points = an_element["points"]
                                    round = an_element["round"]
                                    end_rel = arrow_points[-1]
                                    start_rel = arrow_points[0]
                                    middle_points_list = arrow_points[1:-1]

                                    paths_list = get_arrow_points(
                                        0, 0, start_rel, end_rel, arrowhead_length=70,
                                        arrowhead_width=40, middle_points_list=middle_points_list, round=round)

                                elif an_element_type == "rectangle":
                                    base_x = an_element["x"]
                                    base_y = an_element["y"]
                                    width = an_element["width"]
                                    height = an_element["height"]
                                    angle = an_element["angle"]
                                    round = an_element["round"]
                                    paths_list = get_rectangle_points(base_x, base_y, width, height, angle=angle, round=round)

                                elif an_element_type == "ellipse":
                                    base_x = an_element["x"]
                                    base_y = an_element["y"]
                                    width = an_element["width"]
                                    height = an_element["height"]
                                    angle = an_element["angle"]
                                    paths_list = get_ellipse_points(base_x, base_y, width, height, angle=angle)

                                elif an_element_type == "diamond":
                                    base_x = an_element["x"]
                                    base_y = an_element["y"]
                                    width = an_element["width"]
                                    height = an_element["height"]
                                    angle = an_element["angle"]
                                    paths_list = get_diamond_points(base_x, base_y, width, height, angle=angle)

                                # I know this is bad, but my brain is fried.... TODO: Redo entire logic
                                if an_element_type == 'arrow':
                                    paths_list = densify_stroke(list(paths_list[0]), factor=pysn.XC_SHAPE_DENSITY, arrows=(an_element_type == 'arrow'))
                                    if dashed:
                                        paths_list = densify_stroke(list(paths_list[0]), factor=pysn.XC_SHAPE_DENSITY, dashed=dashed)
                                    elif dotted:
                                        paths_list = densify_stroke(list(paths_list[0]), factor=pysn.XC_SHAPE_DENSITY, dotted=dotted)
                                    else:
                                        paths_list = densify_stroke(list(paths_list[0]))
                                else:
                                    paths_list = densify_stroke(list(paths_list[0]), factor=pysn.XC_SHAPE_DENSITY)
                                    if dashed:
                                        paths_list = densify_stroke(list(paths_list[0]), dashed=dashed)
                                    elif dotted:
                                        paths_list = densify_stroke(list(paths_list[0]), dotted=dotted)

                                # TODO: rewrite this section. Really bad
                                if a_color == 157:
                                    pen_color = 'dark gray'
                                elif a_color == 201:
                                    pen_color = 'light gray'
                                elif a_color == 0:
                                    pen_color = 'black'
                                else:
                                    pen_color = 'white'

                                vec_points = pysn.vectors2psd(
                                    pysn.build_contours(list(paths_list)),
                                    series=series, pen_size=pen_size, pen_color=pen_color)

                                a_list_ps = vec_points['strokes']

                                if page_nb_str in full_dict_ps:
                                    current_full_dict_ps_page = full_dict_ps[page_nb_str]
                                    current_full_dict_ps_page['strokes_nb'] = current_full_dict_ps_page['strokes_nb'] + len(a_list_ps)
                                    current_full_dict_ps_page['strokes'].extend(a_list_ps)
                                    full_dict_ps[page_nb_str] = current_full_dict_ps_page
                                else:
                                    full_dict_ps[page_nb_str] = {
                                        "strokes_nb": len(a_list_ps),
                                        "strokes": a_list_ps}

                    print()
                    print('   > Building Supernote notebook...')

                    pysn.psdict_to_note(full_dict_ps, titles_dict, filename+'.note', templates_dict=templates_dict)
    except Exception as e:
        print()
        print(f'*** SNEX: {e}')
    finally:
        if new_user_settings:
            new_user_settings['filename'] = "demo/xc2sn/version1.06.excalidraw"
            pysn.save_user_settings(new_user_settings)


if __name__ == "__main__":
    main()
