

import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.Driver;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;

import java.util.Arrays;

public class readGeoInfo {
    /**
     * Created by ubuntu on 2015/7/10 0010.
     */
    public static void main(String[] args) {
        //String fileName_tif = "F:\\java_wk\\readTiff\\Band\\B2.tif";
        String fileName_tif = "H:\\java_wk\\remote_sensing_data\\rm\\fenge\\fangkuai_81_144.tif";
        //String fileName_tif = "F:\\yaogan\\remote_sensing_data\\uv\\uvs\\1.tif";
        gdal.AllRegister();
        Dataset hDataset = gdal.Open(fileName_tif, gdalconstConstants.GA_ReadOnly);
        if (hDataset == null) {
            System.err.println("GDALOpen failed - " + gdal.GetLastErrorNo());
            System.err.println(gdal.GetLastErrorMsg());
            System.exit(1);
        }
        Driver hDriver = hDataset.GetDriver();
        System.out.println("Driver: "+hDriver.getShortName() + "/"+hDriver.getLongName());
        int iXSize = hDataset.getRasterXSize();
        int iYSize = hDataset.getRasterYSize();
        int b = hDataset.getRasterCount();
        double[] GGT= new double[6];
        hDataset.GetGeoTransform(GGT);
        System.out.println("Size is " + iXSize + ", "+iYSize);
        System.out.println("count is " + b);
        System.out.println("GGT is " + Arrays.toString(GGT));
        Band band = hDataset.GetRasterBand(1);
        Band band2 = hDataset.GetRasterBand(2);
        Band band3 = hDataset.GetRasterBand(3);
        System.out.println("type is " + band.GetRasterDataType());
        int[] buf = new int[2500];
        int[] buf2 = new int[6];
        int[] buf3 = new int[6];
        int xoff=0;
        int yoff=0;
        band.ReadRaster(xoff, yoff, 50, 50, buf);
        //band2.ReadRaster(xoff, yoff, 60, 60, buf2);
        //band3.ReadRaster(xoff, yoff, 60, 60, buf3);
        System.out.println(buf.length);
        System.out.println(Arrays.toString(buf));
        //for (int i=0;i<60;i++){
        //    int[] num=new int[50];
        //    num=Arrays.copyOfRange(buf,60*i,60*(i+1));
        //    System.out.println(Arrays.toString(num));
        //}
        //System.out.println(Arrays.toString(buf2));
        //System.out.println(Arrays.toString(buf3));

        /*int[] buf = new int[iXSize];
        for (int i = 0; i < 10; i++) {
            band.ReadRaster(500, 500, 10, 10, buf);
            for (int j = 0; j < 10; j++)
                System.out.print(buf[j] + ", ");
            System.out.println("\n");
        }*/
        hDataset.delete();
    }
}
