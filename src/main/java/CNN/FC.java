package CNN;


class FC{                       //有问题?，没做完?
    float[] vector;
    Array_Kernel KernelCount;    //卷积核数量，决定下一隐藏层的通道数
    float[] bias;


    void biasIniting(int bias_size){
        bias=new float[bias_size];
        java.util.Random random = new java.util.Random();
        for (int i=0;i<bias_size;i++){
            float ff=0;
            while(ff<=0 || ff>1){
                ff=(float)random.nextDouble();
            }
            this.bias[i]=ff;
        }
    }


    FC(){
        //System.out.println("please input PM or FC and proper parameter");
    }

    void progress_PM(PoolMat PM){
        for (int i=0;i<KernelCount.length;i++) {
            float n=0;
            for (int band = 0; band < PM.nBandCount; band++) {
                for (int y = 0; y < PM.ySize; y++) {
                    for (int x = 0; x < PM.xSize; x++) {
                        n += (KernelCount.AK[i][band][y][x] * PM.mat[band][y][x]);
                    }
                }
            }
            vector[i]= n+ bias[i];
        }
    }

    FC(PoolMat PM, int[] AK_size){
        int[] ak_size={AK_size[0],AK_size[1],PM.ySize};
        KernelCount= new Array_Kernel(ak_size);
        biasIniting(KernelCount.length);
        vector=new float[KernelCount.length];

        progress_PM(PM);
    }

    FC(PoolMat PM, float[][][][] AK,float[] bias){
        KernelCount=new Array_Kernel(AK);
        this.bias=bias;
        vector=new float[KernelCount.length];

        progress_PM(PM);
    }

    void progress_FC(FC fc){
        for (int i=0;i<KernelCount.length;i++) {
            float n=0;
            for (int band=0;band<fc.vector.length;band++){
                n +=KernelCount.AK[i][band][0][0]*fc.vector[band];              //bias是否共享，如果共享则直接加共享值
            }
            vector[i]=n+bias[i];
        }
    }

    FC(FC fc,int[] AK_size){
        int[] ak_size={AK_size[0],AK_size[1],1};
        KernelCount= new Array_Kernel(ak_size);
        biasIniting(KernelCount.length);
        vector=new float[KernelCount.length];

        progress_FC(fc);
    }

    FC(FC fc,float[][][][] AK,float[] bias){
        KernelCount=new Array_Kernel(AK);
        this.bias=bias;
        vector=new float[KernelCount.length];

        progress_FC(fc);
    }

    void norm(){
        float min=8192;
        float max=0;
        float span=0;
        for (int band=0;band<vector.length;band++){
            float k=vector[band];
            if (k<min){min=k;}
            if (k>max){max=k;}
        }
        span=max-min;
        float[] vec=new float[vector.length];
        for (int band=0;band<vector.length;band++){
            vec[band]=((vector[band]-min)/span);
        }
        vector=vec;
    }

    void relu(){
        for (int band=0;band<vector.length;band++){
            if (vector[band]<=0){
                vector[band]=0;
            }
            else{
                continue;
            }
        }
    }
}
