package CNN;



class OutputLayer extends FC{                   //只是多了print

    void percentify(){
        float sum=0;

        for(int band=0;band<vector.length;band++){
            sum+=vector[band];
        }

        float[] vec=new float[vector.length];
        for (int band=0;band<vector.length;band++){
            vec[band]=vector[band]/sum;
        }

        vector=vec;
    }

    void softmax(){
        double sum=0;
        double[] newvec=new double[vector.length];
        for (int band=0;band<vector.length;band++){
            newvec[band]=Math.pow(Math.E,vector[band]);
            sum+= newvec[band];
        }
        for (int band=0;band<vector.length;band++){
            //newvec[band]/=sum;
            vector[band]=(float)(newvec[band]/sum);
        }
    }


    OutputLayer(PoolMat PM, int[] AK_size){
        super(PM,AK_size);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    OutputLayer(PoolMat PM, float[][][][] AK,float[] bias){
        super(PM,AK,bias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    OutputLayer(FC fc,int[] AK_size){
        super(fc,AK_size);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    OutputLayer(FC fc,float[][][][] AK,float[] bias){
        super(fc,AK,bias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }
}

