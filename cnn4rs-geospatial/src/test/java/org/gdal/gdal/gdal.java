package org.gdal.gdal;

/** Test-scope GDAL binding contract stub; it is not packaged with production code. */
public final class gdal {
    private static Dataset dataset;
    private static int allRegisterCalls;
    private static int lastOpenAccess;
    private static boolean openedBeforeRegistration;

    private gdal() {
    }

    public static void reset() {
        dataset = new Dataset();
        allRegisterCalls = 0;
        lastOpenAccess = Integer.MIN_VALUE;
        openedBeforeRegistration = false;
    }

    public static void AllRegister() {
        allRegisterCalls++;
    }

    public static Dataset Open(String path, int access) {
        openedBeforeRegistration = allRegisterCalls == 0;
        lastOpenAccess = access;
        return dataset;
    }

    public static int allRegisterCalls() {
        return allRegisterCalls;
    }

    public static int lastOpenAccess() {
        return lastOpenAccess;
    }

    public static boolean openedBeforeRegistration() {
        return openedBeforeRegistration;
    }

    public static int lastReadDataType() {
        return dataset.band().lastDataType();
    }

    public static int lastReadStatus() {
        return dataset.band().lastStatus();
    }

    public static void setReadStatus(int status) {
        dataset.band().setStatus(status);
    }

    public static boolean datasetWasDeleted() {
        return dataset.deleted();
    }

    static {
        reset();
    }
}
