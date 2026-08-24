package io.github.chjers.cnn4rs.geospatial;

import io.github.chjers.cnn4rs.core.Tensor;
import java.nio.file.Path;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GdalAvailabilityTest {
    @Test
    void explainsHowToEnableGdalWhenBindingsAreAbsent() {
        GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
                () -> new GdalRasterReader(new MissingGdalClassLoader()).read(Path.of("missing.tif")));

        assertTrue(error.getMessage().contains("org.gdal"));
    }

    @Test
    void explainsHowToEnableGdalWhenConstantsBindingsAreAbsent() {
        gdal.reset();

        GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
                () -> new GdalRasterReader(new MissingConstantsClassLoader()).read(Path.of("missing.tif")));

        assertTrue(error.getMessage().contains("gdalconstConstants"));
        assertTrue(error.getCause() instanceof ClassNotFoundException);
    }

    @Test
    void explainsHowToEnableGdalWhenRegistrationCannotLinkNativeLibraries() {
        gdal.reset();
        UnsatisfiedLinkError nativeBindingFailure = new UnsatisfiedLinkError("missing gdalalljni");
        gdal.setAllRegisterFailure(nativeBindingFailure);

        GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
                () -> new GdalRasterReader().read(Path.of("contract.tif")));

        assertEquals(nativeBindingFailure, error.getCause());
        assertTrue(error.getMessage().contains("AllRegister"));
    }

    @Test
    void explainsHowToEnableGdalWhenOpenCannotLinkNativeLibraries() {
        gdal.reset();
        UnsatisfiedLinkError nativeBindingFailure = new UnsatisfiedLinkError("missing gdalalljni");
        gdal.setOpenFailure(nativeBindingFailure);

        GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
                () -> new GdalRasterReader().read(Path.of("contract.tif")));

        assertEquals(nativeBindingFailure, error.getCause());
        assertTrue(error.getMessage().contains("Open"));
    }

    @Test
    void registersGdalAndReadsFloatBufferThroughBindingContract() {
        gdal.reset();

        RasterTile tile = new GdalRasterReader().read(Path.of("contract.tif"));

        assertEquals(1, gdal.allRegisterCalls());
        assertTrue(!gdal.openedBeforeRegistration());
        assertEquals(gdalconstConstants.GA_ReadOnly, gdal.lastOpenAccess());
        assertEquals(gdalconstConstants.GDT_Float32, gdal.lastReadDataType());
        assertEquals(0, gdal.lastReadStatus());
        assertEquals(1f, tile.data().get(0, 0, 0));
        assertEquals(4f, tile.data().get(0, 1, 1));
        assertTrue(gdal.datasetWasDeleted());
    }

    @Test
    void rejectsNonzeroGdalReadRasterStatus() {
        gdal.reset();
        gdal.setReadStatus(7);

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> new GdalRasterReader().read(Path.of("contract.tif")));

        assertTrue(error.getMessage().contains("status 7"));
        assertFalse(error instanceof GdalUnavailableException);
        assertTrue(gdal.datasetWasDeleted());
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

    private static final class MissingGdalClassLoader extends ClassLoader {
        private MissingGdalClassLoader() {
            super(GdalRasterReader.class.getClassLoader());
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            if (name.equals("org.gdal.gdal.gdal")) {
                throw new ClassNotFoundException(name);
            }
            return super.loadClass(name, resolve);
        }
    }

    private static final class MissingConstantsClassLoader extends ClassLoader {
        private MissingConstantsClassLoader() {
            super(GdalRasterReader.class.getClassLoader());
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            if (name.equals("org.gdal.gdalconst.gdalconstConstants")) {
                throw new ClassNotFoundException(name);
            }
            return super.loadClass(name, resolve);
        }
    }
}
