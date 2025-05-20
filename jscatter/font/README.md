# Fonts for Labeling

To efficiently compute the width and height of labels, we're creating a spec
file from signed distance fields text rendering. By default we're supporting
Arial (regular, bold, italic, and bold italic) but you can generate your own
spec file with [`msdf-bmfont-xml`](https://github.com/soimy/msdf-bmfont-xml):

```sh
$ npm i -g msdf-bmfont-xml
$ msdf-bmfont -f json -s 24 -t sdf --smart-size my_font.ttf
```

Once you have a spec file you can create your own font as follows:

```py
from jscatter import Scatter, Font

my_font = Font('./path/to/my_font.json')
```
