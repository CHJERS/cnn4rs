package org.gdal.gdal;

/** Test-scope binding contract exposing GDAL's float-buffer ReadRaster overload. */
public final class Band {
    private int lastDataType = Integer.MIN_VALUE;
    private int lastStatus;
    private int status;

    public int ReadRaster(int xOffset, int yOffset, int width, int height, int dataType,
                          float[] buffer) {
        lastDataType = dataType;
        if (status == 0) {
            System.arraycopy(new float[] {1f, 2f, 3f, 4f}, 0, buffer, 0, buffer.length);
        }
        lastStatus = status;
        return status;
    }

    void setStatus(int status) {
        this.status = status;
    }

    int lastDataType() {
        return lastDataType;
    }

    int lastStatus() {
        return lastStatus;
    }
}
