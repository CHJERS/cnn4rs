package io.github.chjers.cnn4rs.geospatial;

/** Indicates that GDAL's Java bindings or matching native libraries are unavailable. */
public final class GdalUnavailableException extends IllegalStateException {
    public GdalUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
