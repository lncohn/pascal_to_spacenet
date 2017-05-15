This is a GitHub repository containing python scripts to convert Pascal VOC SBD labels to geoJson Labels for the SpaceNet competition. Note that this code works for the first SpaceNet competition over Rio de Janeiro.

Run the command:

python pascal_to_spacenet.py jpg_dir tif_dir geojson_dir updated_geojson_dir

in a directory containing a subdirectory of raster tif files, a subdirectory of raster jpg files, an empty subdirectory for geojson file outputs, and another empty subdirectory for further processed geojson file outputs

The resulting geojson output labels will be in updated_geojson_dir.
