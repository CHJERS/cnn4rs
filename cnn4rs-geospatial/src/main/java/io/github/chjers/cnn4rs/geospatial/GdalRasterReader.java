package io.github.chjers.cnn4rs.geospatial;

import io.github.chjers.cnn4rs.core.Tensor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.util.Objects;

/**
 * A read-only raster reader that uses GDAL only when its Java bindings and native libraries
 * are supplied at runtime.
 */
public final class GdalRasterReader implements RasterReader {
    private static final String GDAL_CLASS = "org.gdal.gdal.gdal";
    private static final String GDAL_CONSTANTS_CLASS = "org.gdal.gdalconst.gdalconstConstants";
    private final ClassLoader gdalClassLoader;

    public GdalRasterReader() {
        this(GdalRasterReader.class.getClassLoader());
    }

    GdalRasterReader(ClassLoader gdalClassLoader) {
        this.gdalClassLoader = Objects.requireNonNull(gdalClassLoader, "GDAL class loader must not be null");
    }

    @Override
    public RasterTile read(Path path) {
        Objects.requireNonNull(path, "raster path must not be null");
        Class<?> gdal = loadGdal();
        Object dataset = null;
        try {
            register(gdal);
            dataset = openReadOnly(gdal, path);
            int width = invokeInt(dataset, "GetRasterXSize");
            int height = invokeInt(dataset, "GetRasterYSize");
            int bandCount = invokeInt(dataset, "GetRasterCount");
            if (width <= 0 || height <= 0 || bandCount <= 0) {
                throw new IllegalStateException("GDAL dataset has invalid raster dimensions");
            }
            return new RasterTile(Tensor.of(readBands(dataset, width, height, bandCount,
                    gdalConstant("GDT_Float32")),
                    bandCount, height, width), 0, 0);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException("unable to read raster through GDAL's read-only Java API", exception);
        } finally {
            destroy(dataset);
        }
    }

    private Class<?> loadGdal() {
        try {
            return Class.forName(GDAL_CLASS, true, gdalClassLoader);
        } catch (ClassNotFoundException | LinkageError exception) {
            throw new GdalUnavailableException("GDAL is unavailable: add the " + GDAL_CLASS
                    + " Java bindings and matching native GDAL libraries to the runtime.", exception);
        }
    }

    private static void register(Class<?> gdal) throws ReflectiveOperationException {
        gdal.getMethod("AllRegister").invoke(null);
    }

    private Object openReadOnly(Class<?> gdal, Path path) throws ReflectiveOperationException {
        int readOnly = gdalConstant("GA_ReadOnly");
        Method open = gdal.getMethod("Open", String.class, int.class);
        Object dataset = open.invoke(null, path.toString(), readOnly);
        if (dataset == null) {
            throw new IllegalArgumentException("GDAL could not open raster " + path + " for read-only access");
        }
        return dataset;
    }

    private int gdalConstant(String name) throws ReflectiveOperationException {
        Class<?> constants = Class.forName(GDAL_CONSTANTS_CLASS, true, gdalClassLoader);
        Field field = constants.getField(name);
        Object value = field.get(null);
        if (!(value instanceof Number number)) {
            throw new IllegalStateException("GDAL constant " + name + " is not numeric");
        }
        return number.intValue();
    }

    private static float[] readBands(Object dataset, int width, int height, int bandCount, int float32)
            throws ReflectiveOperationException {
        int pixelsPerBand = Math.multiplyExact(width, height);
        float[] values = new float[Math.multiplyExact(pixelsPerBand, bandCount)];
        for (int bandIndex = 0; bandIndex < bandCount; bandIndex++) {
            Object band = invoke(dataset, "GetRasterBand", bandIndex + 1);
            if (band == null) {
                throw new IllegalStateException("GDAL returned no raster band " + (bandIndex + 1));
            }
            float[] bandValues = readBand(band, width, height, float32);
            if (bandValues.length != pixelsPerBand) {
                throw new IllegalStateException("GDAL returned " + bandValues.length + " values for raster band "
                        + (bandIndex + 1) + ", expected " + pixelsPerBand);
            }
            System.arraycopy(bandValues, 0, values, bandIndex * pixelsPerBand, pixelsPerBand);
        }
        return values;
    }

    private static float[] readBand(Object band, int width, int height, int float32)
            throws ReflectiveOperationException {
        Method readRaster = band.getClass().getMethod("ReadRaster", int.class, int.class,
                int.class, int.class, int.class, float[].class);
        float[] values = new float[Math.multiplyExact(width, height)];
        Object status = readRaster.invoke(band, 0, 0, width, height, float32, values);
        if (!(status instanceof Integer result)) {
            throw new IllegalStateException("GDAL ReadRaster did not return an integer status");
        }
        if (result != 0) {
            throw new IllegalStateException("GDAL ReadRaster failed with status " + result);
        }
        return values;
    }

    private static int invokeInt(Object target, String methodName) throws ReflectiveOperationException {
        Object value = invoke(target, methodName);
        if (!(value instanceof Number number)) {
            throw new IllegalStateException("GDAL method " + methodName + " did not return a number");
        }
        return number.intValue();
    }

    private static Object invoke(Object target, String methodName, Object... arguments)
            throws ReflectiveOperationException {
        Class<?>[] parameterTypes = new Class<?>[arguments.length];
        for (int index = 0; index < arguments.length; index++) {
            parameterTypes[index] = arguments[index] instanceof Integer ? int.class : arguments[index].getClass();
        }
        try {
            return target.getClass().getMethod(methodName, parameterTypes).invoke(target, arguments);
        } catch (InvocationTargetException exception) {
            throw new IllegalStateException("GDAL method " + methodName + " failed", exception.getCause());
        }
    }

    private static void destroy(Object dataset) {
        if (dataset == null) {
            return;
        }
        try {
            Method delete = dataset.getClass().getMethod("delete");
            if (!Modifier.isStatic(delete.getModifiers())) {
                delete.invoke(dataset);
            }
        } catch (ReflectiveOperationException ignored) {
            // Dataset cleanup is best effort and must not hide the original read failure.
        }
    }
}
