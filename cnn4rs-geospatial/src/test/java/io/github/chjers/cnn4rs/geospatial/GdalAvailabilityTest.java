package io.github.chjers.cnn4rs.geospatial;

import io.github.chjers.cnn4rs.core.Tensor;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GdalAvailabilityTest {
    @Test
    void explainsHowToEnableGdalWhenBindingsAreAbsent() {
        GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
                () -> new GdalRasterReader().read(Path.of("missing.tif")));

        assertTrue(error.getMessage().contains("org.gdal"));
    }

    @Test
    void rasterTileRequiresChwDataAndNonNegativeOffsets() {
        RasterTile tile = new RasterTile(Tensor.zeros(2, 3, 4), 5, 6);

        assertEquals(5, tile.xOffset());
        assertEquals(6, tile.yOffset());
        assertDoesNotThrow(() -> new RasterTile(Tensor.zeros(1, 1, 1), 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new RasterTile(Tensor.zeros(2, 3), 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new RasterTile(Tensor.zeros(1, 1, 1), -1, 0));
        assertThrows(IllegalArgumentException.class, () -> new RasterTile(Tensor.zeros(1, 1, 1), 0, -1));
    }
}
