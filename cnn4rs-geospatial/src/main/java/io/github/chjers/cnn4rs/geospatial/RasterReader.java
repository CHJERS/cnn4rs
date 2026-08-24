package io.github.chjers.cnn4rs.geospatial;

import java.nio.file.Path;

/** Reads one raster tile from a filesystem path. */
public interface RasterReader {
    RasterTile read(Path path);
}
