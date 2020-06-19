
import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;

import java.io.*;
import java.util.Arrays;

public class tiffTransSer {

    static void tiffToSer(String sourcefile,String targetfile) throws IOException {
        float[][][] mat;
        gdal.AllRegister();
        String fileUrl = sourcefile;
        // 读取影像数据
        Dataset dataset = gdal.Open(fileUrl, gdalconstConstants.GA_ReadOnly);
        if (dataset == null) {
            System.err.println("GDALOpen failed - " + gdal.GetLastErrorNo());
            System.err.println(gdal.GetLastErrorMsg());
            System.exit(1);
        }
        // 读取影像信息 宽、高、波段数
        int xSize = dataset.getRasterXSize();
        int ySize = dataset.getRasterYSize();
        int nBandCount = dataset.getRasterCount();

        mat=new float[nBandCount][ySize][xSize];

        int[][] txt=new int[nBandCount][xSize*ySize];

        for (int i = 1; i <= 1; i++) {
            Band band = dataset.GetRasterBand(i);
            int[] cache = new int[xSize*ySize];
            band.ReadRaster(0, 0, xSize, ySize, cache);
            for (int y=0;y<ySize;y++){
                for(int x=0;x<xSize;x++){
                    mat[i-1][y][x]=(float)cache[y*ySize+x];
                }
            }
        }

        FileOutputStream fileOut = new FileOutputStream(targetfile);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(mat);
        out.flush();
        out.close();
        fileOut.close();

    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        /*String sourceflie="C://Temp//SL//L_Z//fangkuai_1791_722.tif";
        String targetfile="C://Temp//SL//fangkuai_1791_722.ser";
        tiffToSer(sourceflie,targetfile);

        FileInputStream fileIn = new FileInputStream("C://Temp//SL//fangkuai_1791_722.ser");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        float[][][] mat = (float[][][]) in.readObject();
        in.close();
        fileIn.close();*/


        /*File f_l = new File("C://Temp//SL//L_Z//");

        String[] ld = f_l.list();
        File f_s = new File("C://Temp//SL//S_Z//");
        String[] st = f_s.list();

        String dir_ld_ser="C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//L//";
        String dir_st_ser="C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//S//";


        for (String l : ld){
            String ld_tif="C://Temp//SL//L_Z//"+l;
            String ld_ser=dir_ld_ser+l.split("\\.")[0]+".ser";
            tiffToSer(ld_tif,ld_ser);
        }

        for (String s : st){
            String ld_tif="C://Temp//SL//S_Z//"+s;
            String ld_ser=dir_st_ser+s.split("\\.")[0]+".ser";
            tiffToSer(ld_tif,ld_ser);
        }*/


        tiffToSer("H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//tiff//train//zhibei//fangkuai_426_421.tif","C://Temp//zb.ser");




    }
}
