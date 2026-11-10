# Input and output

Readers for YOLO segmentation label files and class names. Writing is done
with methods of the data model:
[`Collection.to_txt`][hires.Collection.to_txt],
[`Collection.to_csv`][hires.Collection.to_csv],
[`Collection.save_crops`][hires.Collection.save_crops] and their
[`Album`][hires.Album] counterparts. See
[Reading and writing](../user-guide/io.md) for the file format.

## Readers

::: hires.read_txt

::: hires.read_album

::: hires.read_class_names

## Low-level helpers (`hires.io`)

The functions below are used internally and are public for code that needs
to find or decode images the same way HiReS does.

::: hires.io.IMAGE_EXTENSIONS

::: hires.io.find_images

::: hires.io.read_image

::: hires.io.read_image_size
