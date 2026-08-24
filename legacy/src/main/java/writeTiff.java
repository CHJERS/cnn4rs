
import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.Driver;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;

import java.util.Arrays;

public class writeTiff {
    public static void main(String[] args) {
        String fileUrl = "H:\\java_wk\\all_new_sz_1m\\SZ_NIR1.tif";
        gdal.AllRegister();
        // 读取影像数据
        Dataset dataset = gdal.Open(fileUrl, gdalconstConstants.GA_ReadOnly);
        if (dataset == null) {
            System.err.println("GDALOpen failed - " + gdal.GetLastErrorNo());
            System.err.println(gdal.GetLastErrorMsg());
            System.exit(1);
        }
        //Driver hDriver=dataset.GetDriver();
        // 读取影像信息 宽、高、波段数
        int xSize = dataset.getRasterXSize();
        int ySzie = dataset.getRasterYSize();
        int nBandCount = dataset.getRasterCount();
        System.out.println(xSize);
        System.out.println(ySzie);
        System.out.println(nBandCount);

        int type = dataset.GetRasterBand(1).GetRasterDataType();
        System.out.println(type);
        // 读取仿射变换参数
        double[] im_geotrans = dataset.GetGeoTransform();
        System.out.println(Arrays.toString(im_geotrans));
        // 读取投影
        String im_proj = dataset.GetProjection();
        System.out.println(im_proj);

        int[] cache = new int[25];
        Band band = dataset.GetRasterBand(1);
        band.ReadRaster(14320, 36162, 5, 5, cache);
        System.out.println(Arrays.toString(cache));


        int ys=ySzie/50;
        int xs=xSize/50;
        Dataset d2 = null;
        for (int y=664;y<674;y++){
            for (int x=1708;x<1761;x++){
                String namestr="fangkuai_"+y+"_"+x+".tif";
                d2 = gdal.GetDriverByName("GTiff").Create("H:\\java_wk\\remote_sensing_data\\sz_1m\\binary\\s_weifen\\"+namestr, 50, 50, nBandCount, type);
                double[] im_geotrans_=im_geotrans.clone();
                im_geotrans_[0]+=(50*x);
                im_geotrans_[3]-=(50*y);
                d2.SetGeoTransform(im_geotrans_);
                d2.SetProjection(im_proj);
                for (int j = 1; j <= nBandCount; j++) {
                    int[] cache = new int[2500];
                    Band band = dataset.GetRasterBand(j);
                    int n_x=x*50;
                    int n_y=y*50;
                    band.ReadRaster(n_x, n_y, 50, 50, cache);
                    Band newBand = d2.GetRasterBand(j);
                    newBand.WriteRaster(0, 0, 50, 50, cache);
                    newBand.FlushCache();
                    d2.delete();
                }
            }
        }

        dataset.delete();
        gdal.GDALDestroyDriverManager();



    }

}
