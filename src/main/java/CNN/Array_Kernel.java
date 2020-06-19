package CNN;

class Array_Kernel{
    float[][][][] AK;
    int length;                         //这一层的band
    int nBandCount;                     //上一层的band
    int size_Kernel;                    //kernel大小，为方型


    Array_Kernel(float[][][][] AK){
        this.length=AK.length;
        this.nBandCount=AK[0].length;
        this.size_Kernel=AK[0][0].length;
        this.AK=AK;
    }

    Array_Kernel(int length,int nBandCount,int size_Kernel){
        this.length=length;
        this.nBandCount=nBandCount;
        this.size_Kernel=size_Kernel;
        this.AK=new float[this.length][this.nBandCount][this.size_Kernel][this.size_Kernel];
        java.util.Random random = new java.util.Random();
        for (int l=0;l<this.length;l++){
            for (int i=0;i<this.nBandCount;i++){
                for (int n=0;n<this.size_Kernel;n++){
                    for (int nn=0;nn<this.size_Kernel;nn++){
                        float ff=0;
                        while(ff<=0 || ff>1){
                            ff=(float)random.nextDouble();
                        }
                        this.AK[l][i][n][nn]=ff;
                    }
                }
            }
        }
    }

    Array_Kernel(int[] AK_size){
        this.length=AK_size[0];
        this.nBandCount=AK_size[1];
        this.size_Kernel=AK_size[2];
        this.AK=new float[this.length][this.nBandCount][this.size_Kernel][this.size_Kernel];
        java.util.Random random = new java.util.Random();
        for (int l=0;l<this.length;l++){
            for (int i=0;i<this.nBandCount;i++){
                for (int n=0;n<this.size_Kernel;n++){
                    for (int nn=0;nn<this.size_Kernel;nn++){
                        float ff=0;
                        while(ff<=0 || ff>1){
                            ff=(float)random.nextFloat();
                        }
                        this.AK[l][i][n][nn]=ff;
                    }
                }
            }
        }
    }
}