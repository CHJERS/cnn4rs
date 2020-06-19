
import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class writeTif_Ser {

    public static void main(String[] args) {
        gdal.AllRegister();
        String fileUrl = "H:\\java_wk\\all_new_sz_1m\\AS1";
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

        int size=20;

        int xs=13769/size;
        int ys=35919/size;

        int xx=13819/size;
        int yx=36408/size;

        Dataset d2 = null;
        for (int y=ys;y<yx;y++){                                        //envi读出的后面的值
            for (int x=xs;x<xx;x++){                                    //envi读出的前面的值
                String namestr="fangkuai_"+y+"_"+x+".tif";
                d2 = gdal.GetDriverByName("GTiff").Create("C:\\Temp\\SL\\S2\\"+namestr, size, size, nBandCount, type);
                double[] im_geotrans_=im_geotrans.clone();
                im_geotrans_[0]+=(size*x);
                im_geotrans_[3]-=(size*y);
                d2.SetGeoTransform(im_geotrans_);
                d2.SetProjection(im_proj);

                int[] cache;
                int[][] txt = new int[nBandCount][];
                for (int j = 1; j <= nBandCount; j++) {
                    cache = new int[size*size];
                    Band band = dataset.GetRasterBand(j);
                    int n_x=x*size;
                    int n_y=y*size;
                    band.ReadRaster(n_x, n_y, size, size, cache);
                    Band newBand = d2.GetRasterBand(j);
                    newBand.WriteRaster(0, 0, size, size, cache);
                    newBand.FlushCache();
                    txt[j-1]=cache;
                }
                d2.delete();

                String targetfile="C:\\Temp\\SL_T\\S2\\"+"fangkuai_"+y+"_"+x+".txt";
                BufferedWriter out = null;
                try {
                    out = new BufferedWriter(new FileWriter(targetfile));
                    for (int i = 0; i < nBandCount; i++) {
                        String ss=Arrays.toString(txt[i]);
                        out.write(ss);
                        out.write("\n");
                    }
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        dataset.delete();
        gdal.GDALDestroyDriverManager();

    }
}


