package org.gdal.gdal;

/** Test-scope GDAL binding contract stub; it is not packaged with production code. */
public final class gdal {
    private static Dataset dataset;
    private static int allRegisterCalls;
    private static int lastOpenAccess;
    private static boolean openedBeforeRegistration;
    private static LinkageError allRegisterFailure;
    private static LinkageError openFailure;

    private gdal() {
    }

    public static void reset() {
        dataset = new Dataset();
        allRegisterCalls = 0;
        lastOpenAccess = Integer.MIN_VALUE;
        openedBeforeRegistration = false;
        allRegisterFailure = null;
        openFailure = null;
    }

    public static void AllRegister() {
        if (allRegisterFailure != null) {
            throw allRegisterFailure;
        }
        allRegisterCalls++;
    }

    public static Dataset Open(String path, int access) {
        if (openFailure != null) {
            throw openFailure;
        }
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

    public static void setAllRegisterFailure(LinkageError failure) {
        allRegisterFailure = failure;
    }

    public static void setOpenFailure(LinkageError failure) {
        openFailure = failure;
    }

    public static boolean datasetWasDeleted() {
        return dataset.deleted();
    }

    static {
        reset();
    }
}
