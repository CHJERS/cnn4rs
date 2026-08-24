package io.github.chjers.cnn4rs.examples;

import io.github.chjers.cnn4rs.core.Models;
import io.github.chjers.cnn4rs.core.Tensor;
import io.github.chjers.cnn4rs.geospatial.GdalRasterReader;
import io.github.chjers.cnn4rs.geospatial.RasterTile;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Reads a raster through the optional GDAL adapter and writes a demonstration class index.
 * This is a wiring example, not a trained remote-sensing model.
 */
public final class GdalClassificationExample {
    private GdalClassificationExample() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: GdalClassificationExample <input-raster> <output-text>");
        }
        classify(Path.of(args[0]), Path.of(args[1]));
    }

    /**
     * Classifies the first two raster values with a deterministic demonstration model.
     * GDAL's Java bindings and matching native libraries must be available at runtime.
     */
    public static void classify(Path inputRaster, Path outputText) throws IOException {
        RasterTile tile = new GdalRasterReader().read(inputRaster);
        float[] values = tile.data().toArray();
        if (values.length < 2) {
            throw new IllegalArgumentException("input raster must contain at least two values");
        }
        Tensor features = Tensor.of(new float[] {values[0], values[1]}, 2);
        int prediction = InMemoryClassifierExample.argMax(
                Models.denseClassifier(2, 2, 42L).predict(features));
        java.nio.file.Files.writeString(outputText, "predictedClass=" + prediction + System.lineSeparator());
    }
}
