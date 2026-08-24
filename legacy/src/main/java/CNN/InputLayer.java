package CNN;

import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;

class InputLayer{
    float[][][] mat;
    int xSize;
    int ySize;
    int nBandCount;
    //int sizeofkn;
    //Kernel[] KNCount;       //卷积核数量，决定下一隐藏层的通道数
    //float[] bias;

    float[][][] rmToMat(String file){
        gdal.AllRegister();
        String fileUrl = file;
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
        this.xSize=xSize;
        this.ySize=ySize;
        this.nBandCount=nBandCount;

        float[][][] mat=new float[nBandCount][ySize][xSize];
        for (int i = 1; i <= nBandCount; i++) {
            Band band = dataset.GetRasterBand(i);
            int[] cache = new int[xSize*ySize];
            band.ReadRaster(0, 0, xSize, ySize, cache);
            for (int j = 0; j < ySize; j++) {
                for (int n = 0; n < xSize; n++) {
                    mat[i-1][j][n]=cache[xSize*j+n];
                }
            }
        }
        dataset.delete();
        gdal.GDALDestroyDriverManager();
        return mat;
    }

    void norm(){
        float[] min=new float[nBandCount];
        float[] span=new float[nBandCount];
        for (int band=0;band<nBandCount;band++){
            float mini=9999999;
            float max=0;
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float k=mat[band][y][x];
                    if (k<mini){mini=k;}
                    if (k>max){max=k;}
                }
            }
            min[band]=mini;
            span[band]=max-mini;
        };
        float[][][] matrix=new float[nBandCount][ySize][xSize];
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    matrix[band][y][x]=((mat[band][y][x]-min[band])/span[band]);
                }
            }
        }
        mat=matrix;
    }

    void dotize(){
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    mat[band][y][x]/=65535;
                }
            }
        }
    }

    InputLayer(String tifffile){
        //String file="H:\\java_wk\\remote_sensing_data\\uv\\uvs\\1.tif";
        this.mat=this.rmToMat(tifffile);
        this.dotize();
    }


    InputLayer(float[][][] mat){
        this.mat=mat;
        this.nBandCount=mat.length;
        this.ySize=mat[0].length;
        this.xSize=mat[0][0].length;
        this.dotize();
    }
}