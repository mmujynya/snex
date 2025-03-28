# SNEX (SN Excalidraw)
Supernote - Excalidraw tools
Exports Supernote notebooks to [Excalidraw](https://excalidraw.com/) format
See [YouTube video](https://youtu.be/p1sAisn_xd4?si=bEZ4xpuXrVyZGTjm)


## CHANGE LOG:
- **Version 1.02**: 
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
Run snex.py. The script takes 1 parameter: 
- the name of the source Supernote notebook


For example:
   ```bash
   python snex.py 'test/stroller.note'
   ```
The output would show:
   ```bash
SNEX Version 1.02
-----------------
Processing file: test/stroller.note
Generated file: test/stroller.excalidraw
   ```

You can then head to [Excalidraw website](https://excalidraw.com/) look at the hamburger menu button, use 'open" menu then browse to load the generated file with the extension '.excalidraw'

