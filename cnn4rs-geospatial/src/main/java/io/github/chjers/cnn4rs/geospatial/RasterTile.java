package io.github.chjers.cnn4rs.geospatial;

import io.github.chjers.cnn4rs.core.Tensor;
import java.util.Objects;

/** A channel-height-width raster tile located at a non-negative pixel offset. */
public record RasterTile(Tensor data, int xOffset, int yOffset) {
    public RasterTile {
        Objects.requireNonNull(data, "raster data must not be null");
        if (data.shape().rank() != 3) {
            throw new IllegalArgumentException("raster data must use CHW rank-three tensor shape");
        }
        if (xOffset < 0 || yOffset < 0) {
            throw new IllegalArgumentException("raster offsets must not be negative");
        }
    }
}
