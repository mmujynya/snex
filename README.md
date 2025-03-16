# SNEX (SN Excalidraw)
Supernote - Excalidraw tools

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
   ```

## USAGE
Run snex.py. The script takes 2 parameters: 
- the name of the source Supernote notebook
- the series ("N5" for Manta or "N6" for Nomad) of the tablet that created the notebook

For example:
   ```bash
   python snex.py 'test/testocr.note' 'N5'
   ```
The output would show:
   ```bash
SNEX Version 1.0
----------------
Processing file: test/testocr.note with series: N5
Generated file: test/testocr.excalidraw
   ```

You can then head to [Excalidraw website](https://excalidraw.com/) look at the hamburger menu button, use 'open" menu then browse to load the generated file with the extension '.excalidraw'

