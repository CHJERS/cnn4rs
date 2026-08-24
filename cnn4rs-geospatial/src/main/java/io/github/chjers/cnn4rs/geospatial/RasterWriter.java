package io.github.chjers.cnn4rs.geospatial;

import java.nio.file.Path;

/** Writes a raster tile to a filesystem path. */
public interface RasterWriter {
    void write(Path path, RasterTile tile);
}
