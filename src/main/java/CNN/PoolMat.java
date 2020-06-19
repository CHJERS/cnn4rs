package CNN;


class PoolMat{
    float[][][] mat;
    int xSize;
    int ySize;
    int nBandCount;

    void pm_max(HiddenLayer HL, int poolSize){
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float n=0;
                    for (int i=0;i<poolSize;i++){
                        for (int ii=0;ii<poolSize;ii++) {
                            if (n < HL.mat[band][y*poolSize+i][x*poolSize+ii]){
                                n=HL.mat[band][y*poolSize+i][x*poolSize+ii];
                            }
                        }
                    }
                    mat[band][y][x]=n;
                }
            }
        }
    }

    void pm_mean(HiddenLayer HL, int poolSize){
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float n=0;
                    for (int i=0;i<poolSize;i++){
                        for (int ii=0;ii<poolSize;ii++) {
                            n+=HL.mat[band][y*poolSize+i][x*poolSize+ii];
                        }
                    }
                    mat[band][y][x]=n/(poolSize*poolSize);
                }
            }
        }
    }

    PoolMat(HiddenLayer HL, int poolSize, String poolType){
        this.xSize=HL.xSize/poolSize;
        this.ySize=HL.ySize/poolSize;
        this.nBandCount=HL.nBandCount;
        this.mat=new float[nBandCount][ySize][xSize];              //暂时未考虑鲁棒性
        if (poolType=="max"){
            pm_max(HL,poolSize);
        }
        else if(poolType=="mean"){
            pm_mean(HL,poolSize);
        }
        else{
            System.out.println("illegal input,please input 'max' or 'mean'.");
        }
    }
}