package org.gdal.gdal;

/** Test-scope stand-in for a one-band, two-by-two GDAL dataset. */
public final class Dataset {
    private final Band band = new Band();
    private boolean deleted;

    public int GetRasterXSize() {
        return 2;
    }

    public int GetRasterYSize() {
        return 2;
    }

    public int GetRasterCount() {
        return 1;
    }

    public Band GetRasterBand(int index) {
        return index == 1 ? band : null;
    }

    public void delete() {
        deleted = true;
    }

    Band band() {
        return band;
    }

    boolean deleted() {
        return deleted;
    }
}
