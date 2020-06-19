package CNN;


class HiddenLayer {
    float[][][] mat;
    int xSize;
    int ySize;
    int nBandCount;
    Array_Kernel KernelCount;    //卷积核数量，决定下一隐藏层的通道数
    //int sizeofkn;
    //Kernel[] KNCount;       //卷积核数量，决定下一隐藏层的通道数
    float[] bias;
    int step;

    float[][][] receiveMat(InputLayer IL){
        return IL.mat;
    }

    float[][][] receiveMat(HiddenLayer HL){
        return HL.mat;
    }

    float[][][] receiveMat(PoolMat PM){return PM.mat;}

    int receiveYsize(InputLayer IL){
        return IL.ySize;
    }

    int receiveYsize(HiddenLayer HL){
        return HL.ySize;
    }

    int receiveYsize(PoolMat PM){ return PM.ySize; }

    int receiveYsize(float[][][] last_mat){ return last_mat[0].length; }

    int receiveXsize(InputLayer IL){
        return IL.xSize;
    }

    int receiveXsize(HiddenLayer HL){
        return HL.xSize;
    }

    int receiveXsize(PoolMat PM){ return PM.xSize; }

    int receiveXsize(float[][][] last_mat){
        return last_mat[0][0].length;
    }

    int receiveNband(InputLayer IL){
        return IL.nBandCount;
    }

    int receiveNband(HiddenLayer HL){
        return HL.nBandCount;
    }

    int receiveNband(PoolMat PM){
        return PM.nBandCount;
    }

    int receiveNband(float[][][] last_mat){
        return last_mat.length;
    }

    float[][][] define_pa(int L_ySize,int L_xSize,int L_nBandCount, float[][][] L_mat){               //defind past mat
        float[][][] mat;
        int mat_ysize;
        int mat_xsize;
        int yHL;
        int xHL;

        int yremainder=(L_ySize-this.KernelCount.size_Kernel)%step;

        if (yremainder>0){
            yHL=(L_ySize-this.KernelCount.size_Kernel)/step+2;
        }
        else{
            yHL=(L_ySize-this.KernelCount.size_Kernel)/step+1;
        }
        mat_ysize=yHL*step+(this.KernelCount.size_Kernel-step);


        int xremainder=(L_xSize-this.KernelCount.size_Kernel)%step;

        if (xremainder>0){
            xHL=(L_xSize-this.KernelCount.size_Kernel)/step+2;
        }
        else{
            xHL=(L_xSize-this.KernelCount.size_Kernel)/step+1;
        }

        mat_xsize=xHL*step+(this.KernelCount.size_Kernel-step);

        if(yremainder>0 && xremainder>0){
            mat=new float[L_nBandCount][mat_ysize][mat_xsize];
            for (int band=0;band<L_nBandCount;band++){
                for (int y=0;y<mat_ysize;y++){
                    for (int x=0;x<mat_xsize;x++){
                        if (x<L_xSize && y<L_ySize){
                            mat[band][y][x]=L_mat[band][y][x];
                        }
                        else{
                            mat[band][y][x]=0;
                        }
                    }
                }

            }
        }
        else{
            mat=L_mat;
        }

        this.ySize=yHL;
        this.xSize=xHL;
        this.nBandCount=KernelCount.length;
        this.mat=new float[KernelCount.length][this.ySize][this.xSize];
        return mat;
    }


    void receiveYXBM(int L_ySize,int L_xSize,int L_nBandCount, float[][][] L_mat ){
        float[][][] mat=define_pa(L_ySize,L_xSize,L_nBandCount,L_mat);
        for (int i=0;i<KernelCount.length;i++){
            for (int yhl=0;yhl<this.ySize;yhl++){
                for (int xhl=0;xhl<this.xSize;xhl++){
                    float n=0;
                    for (int band=0;band<KernelCount.nBandCount;band++){
                        for (int y=0;y<KernelCount.size_Kernel;y++){
                            for (int x=0;x<KernelCount.size_Kernel;x++){
                                n+=((KernelCount.AK[i][band][y][x]*mat[band][yhl*step+y][xhl*step+x]));
                            }
                        }
                    }
                    this.mat[i][yhl][xhl]=n+bias[i];
                }
            }
        }
    }

    void receive(InputLayer IL){
        int L_ySize=receiveYsize(IL);
        int L_xSize=receiveXsize(IL);
        int L_nBandCount=receiveNband(IL);
        float[][][] L_mat=receiveMat(IL);

        receiveYXBM(L_ySize,L_xSize,L_nBandCount,L_mat);
    }

    void receive(float[][][] last_mat){
        int L_ySize=receiveYsize(last_mat);
        int L_xSize=receiveXsize(last_mat);
        int L_nBandCount=receiveNband(last_mat);
        float[][][] L_mat=last_mat;

        receiveYXBM(L_ySize,L_xSize,L_nBandCount,L_mat);
    }

    void receive(HiddenLayer HL){
        int L_ySize=receiveYsize(HL);
        int L_xSize=receiveXsize(HL);
        int L_nBandCount=receiveNband(HL);
        float[][][] L_mat=receiveMat(HL);

        receiveYXBM(L_ySize,L_xSize,L_nBandCount,L_mat);
    }

    void receive(PoolMat PM){
        int L_ySize=receiveYsize(PM);
        int L_xSize=receiveXsize(PM);
        int L_nBandCount=receiveNband(PM);
        float[][][] L_mat=receiveMat(PM);

        receiveYXBM(L_ySize,L_xSize,L_nBandCount,L_mat);
    }

    void biasIniting(int bias_size){
        bias=new float[bias_size];
        java.util.Random random = new java.util.Random();
        for (int i=0;i<bias_size;i++){
            float ff=0;
            while(ff<=0 || ff>1){
                ff=random.nextFloat();
            }
            this.bias[i]=ff;
        }
    }

    HiddenLayer(){}

    HiddenLayer(InputLayer IL, int step, int[] AK_size){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        receive(IL);
    }

    HiddenLayer(float[][][] IL_mat, int step, int[] AK_size){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        receive(IL_mat);
    }



    HiddenLayer(HiddenLayer HL, int step, int[] AK_size){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        receive(HL);
    }

    HiddenLayer(PoolMat PM, int step, int[] AK_size ){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        receive(PM);
    }

    HiddenLayer(InputLayer IL, int step, float[][][][] nkn, float[] bias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        receive(IL);
    }

    HiddenLayer(float[][][] IL_mat, int step, float[][][][] nkn, float[] bias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        receive(IL_mat);
    }

    HiddenLayer(HiddenLayer HL, int step, float[][][][] nkn, float[] bias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        receive(HL);
    }

    HiddenLayer(PoolMat PM, int step, float[][][][] nkn, float[] bias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        receive(PM);
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

    void relu(){
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    if (mat[band][y][x]<=0){
                        mat[band][y][x]=0;
                    }
                    else{
                        continue;
                    }
                }
            }
        }
    }
}

