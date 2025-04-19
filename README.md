# SNEX (SN Excalidraw)
Supernote - Excalidraw tools
Exports Supernote notebooks to [Excalidraw](https://excalidraw.com/) format
See [YouTube video](https://youtu.be/p1sAisn_xd4?si=bEZ4xpuXrVyZGTjm)


## CHANGE LOG:
- **Version 1.06**:
  - Importing images from Excalidraw to Supernote
  - Basic command line interface
- Version 1.05:
  - Importing from Excalidraw to Supernote

- Version 1.02: 
  - The series (N5 for Manta or N6 for other) is inferred from the file content
  - Frames are locked by default


## NOTICE & Credit: 

This code uses and alters [Github supernote-tool](https://github.com/jya-dev/supernote-tool/tree/master)
See changes at: [Supernotelib commits](https://gitlab.com/mmujynya/pysn-digest/-/commit/c8b9ca72c71293a666176405e1bc1fc21e90e0ba)

It also uses functions from [PySN (python for Supernote)](https://gitlab.com/mmujynya/pysn-digest)

## REQUIREMENTS
Python 3.12.2 (Version 3.13.x or later may not work)


## INSTALLATION
1. Open your terminal.
2. Clone the repository:
   ```bash
   git clone https://github.com/mmujynya/snex.git
   cd snex
   pip install -r requirements.txt
   ```

## USAGE
- Command Line Interface help: python snex.py --help
- Create an Excalidraw scene with a few blank pages 
  - This is needed because Excalidraw has an infinite canvas, while the Supernote format relies on pages of a given dimension. We simulate the SN pages by creating frame objects.
  - Command: python snex.py --blank
  - You can change the number of default pages. For example, to have 5 pages, run: python snex.py --pages 5

- Export a notebook to Excalidraw
  - Command: Python snex.py <filename with .note extension>
  - For example: python snex.py 'test/alice.note' will create 'alice.note.excalidraw' in the 'test' subfolder. In Excalidraw, you can load such file from the menu-> open buttons

  Head to [Excalidraw website](https://excalidraw.com/) look at the hamburger menu button, use "open" menu then browse to load the generated file with the extension ".excalidraw"

- Import an Excalidraw file to Supernote
  - Command: python snex.py <filename with .excalidraw extension >



# ISSUES
- When converting text to notes, pay attention to not touching the limits of the page. In particular, stick with the "barlow" font family, because it is narrow enough.
- Lower resolution of the Excalidraw canvas may result into small penstrokes details being lost when eporting a ".note" file. A consequence of that bug is that "dotted" shapes re-uploaded from the Supernote to Excalidraw may disappear (too small fragments).
- No links, no headers, no images

