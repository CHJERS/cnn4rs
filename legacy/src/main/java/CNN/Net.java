package CNN;


import java.util.Arrays;
import java.util.HashMap;

public class Net{

    HashMap d_kernel=new HashMap<Integer,float[][][][]>();
    HashMap d_bias=new HashMap<Integer,float[]>();
    float[] result;


    int[][] WC_size = {{6, 4, 3},
            {16, 6, 3},
            //{21, 13, 2},
            {64,16,0},
            //{64,64,0},
            {2,64,0}
    };

    public Net(String namestr) {
        //long startTime =  System.currentTimeMillis();

        InputLayer IL = new InputLayer(namestr);

        HiddenLayer HL_0 = new HiddenLayer(IL, 1, WC_size[0]);
        d_kernel.put(d_kernel.size(),HL_0.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_0.bias);
        HL_0.relu();
        HL_0.norm();
        //HL_0.relu();
        PoolMat PM_0=new PoolMat(HL_0,2,"max");

        HiddenLayer HL_1 = new HiddenLayer(PM_0, 1, WC_size[1]);
        d_kernel.put(d_kernel.size(),HL_1.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_1.bias);
        HL_1.relu();
        HL_1.norm();
        //HL_1.relu();
        PoolMat PM_1=new PoolMat(HL_1,2,"max");

        /*HiddenLayer HL_2 = new HiddenLayer(PM_1, 1, WC_size[2]);
        d_kernel.put(d_kernel.size(),HL_2.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_2.bias);
        HL_2.relu();
        HL_2.norm();
        //HL_2.relu();
        PoolMat PM_2=new PoolMat(HL_2,2,"max");*/

        FC FC_0=new FC(PM_1,WC_size[2]);
        d_kernel.put(d_kernel.size(),FC_0.KernelCount.AK);
        d_bias.put(d_bias.size(),FC_0.bias);
        FC_0.relu();
        FC_0.norm();
        //FC_0.relu();

        /*FC FC_1=new FC(FC_0,WC_size[4]);
        d_kernel.put(d_kernel.size(),FC_1.KernelCount.AK);
        d_bias.put(d_bias.size(),FC_1.bias);
        FC_1.relu();
        FC_1.norm();*/
        //FC_1.relu();

        OutputLayer OP=new OutputLayer(FC_0,WC_size[3]);
        OP.softmax();
        //OP.percentify();

        System.out.println(Arrays.toString(OP.vector));
        d_kernel.put(d_kernel.size(),OP.KernelCount.AK);
        d_bias.put(d_bias.size(),OP.bias);

        //long spanTime =  (System.currentTimeMillis()-startTime);
        //System.out.println(spanTime);
    }

    public Net(float[][][] IL_mat) {
        //long startTime =  System.currentTimeMillis();

        InputLayer IL = new InputLayer(IL_mat);

        HiddenLayer HL_0 = new HiddenLayer(IL, 1, WC_size[0]);
        d_kernel.put(d_kernel.size(),HL_0.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_0.bias);
        HL_0.relu();
        HL_0.norm();
        //HL_0.relu();
        PoolMat PM_0=new PoolMat(HL_0,2,"max");

        HiddenLayer HL_1 = new HiddenLayer(PM_0, 1, WC_size[1]);
        d_kernel.put(d_kernel.size(),HL_1.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_1.bias);
        HL_1.relu();
        HL_1.norm();
        //HL_1.relu();
        PoolMat PM_1=new PoolMat(HL_1,2,"max");

        /*HiddenLayer HL_2 = new HiddenLayer(PM_1, 1, WC_size[2]);
        d_kernel.put(d_kernel.size(),HL_2.KernelCount.AK);
        d_bias.put(d_bias.size(),HL_2.bias);
        HL_2.relu();
        HL_2.norm();
        //HL_2.relu();
        PoolMat PM_2=new PoolMat(HL_2,2,"max");*/

        FC FC_0=new FC(PM_1,WC_size[2]);
        d_kernel.put(d_kernel.size(),FC_0.KernelCount.AK);
        d_bias.put(d_bias.size(),FC_0.bias);
        FC_0.relu();
        FC_0.norm();
        //FC_0.relu();

        /*FC FC_1=new FC(FC_0,WC_size[4]);
        d_kernel.put(d_kernel.size(),FC_1.KernelCount.AK);
        d_bias.put(d_bias.size(),FC_1.bias);
        FC_1.relu();
        FC_1.norm();*/
        //FC_1.relu();

        OutputLayer OP=new OutputLayer(FC_0,WC_size[3]);
        OP.softmax();
        //OP.percentify();

        System.out.println(Arrays.toString(OP.vector));
        d_kernel.put(d_kernel.size(),OP.KernelCount.AK);
        d_bias.put(d_bias.size(),OP.bias);

        //long spanTime =  (System.currentTimeMillis()-startTime);
        //System.out.println(spanTime);
    }



    public Net(String ILnamestr,HashMap d_kernel,HashMap d_bias){

        this.d_kernel.putAll(d_kernel);
        this.d_bias.putAll(d_bias);

        InputLayer IL = new InputLayer(ILnamestr);

        HiddenLayer HL_0 = new HiddenLayer(IL, 1, (float[][][][])this.d_kernel.get(0),(float[])this.d_bias.get(0));
        HL_0.norm();
        HL_0.relu();
        PoolMat PM_0=new PoolMat(HL_0,2,"mean");

        HiddenLayer HL_1 = new HiddenLayer(PM_0, 1, (float[][][][])this.d_kernel.get(1),(float[])this.d_bias.get(1));
        HL_1.norm();
        HL_1.relu();
        PoolMat PM_1=new PoolMat(HL_1,2,"mean");

        /*HiddenLayer HL_2 = new HiddenLayer(PM_1, 1, (float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2));
        HL_2.norm();
        HL_2.relu();
        PoolMat PM_2=new PoolMat(HL_2,2,"mean");*/

        FC FC_0=new FC(PM_1,(float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2));
        FC_0.norm();
        FC_0.relu();

        /*FC FC_1=new FC(FC_0,(float[][][][])this.d_kernel.get(3),(float[])this.d_bias.get(3));
        FC_1.norm();
        FC_1.relu();*/

        OutputLayer OP=new OutputLayer(FC_0,(float[][][][])this.d_kernel.get(3),(float[])this.d_bias.get(3));
        OP.softmax();
        System.out.println(Arrays.toString(OP.vector));

    }

    public Net(float[][][] IL_mat,HashMap d_kernel,HashMap d_bias){

        this.d_kernel.putAll(d_kernel);
        this.d_bias.putAll(d_bias);

        for (int band=0;band<IL_mat.length;band++){
            for (int y=0;y<IL_mat[0].length;y++){
                for (int x=0;x<IL_mat[0][0].length;x++){
                    IL_mat[band][y][x]/=65535;
                }
            }
        }

        HiddenLayer HL_0 = new HiddenLayer(IL_mat, 1, (float[][][][])this.d_kernel.get(0),(float[])this.d_bias.get(0));
        HL_0.relu();
        HL_0.norm();
        //HL_0.relu();
        PoolMat PM_0=new PoolMat(HL_0,2,"max");

        HiddenLayer HL_1 = new HiddenLayer(PM_0, 1, (float[][][][])this.d_kernel.get(1),(float[])this.d_bias.get(1));
        HL_1.relu();
        HL_1.norm();
        //HL_1.relu();
        PoolMat PM_1=new PoolMat(HL_1,2,"max");

        /*HiddenLayer HL_2 = new HiddenLayer(PM_1, 1, (float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2));
        HL_2.relu();
        HL_2.norm();
        //HL_2.relu();
        PoolMat PM2=new PoolMat(HL_2,2,"max");*/

        FC FC_0=new FC(PM_1,(float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2));
        FC_0.relu();
        FC_0.norm();
        //FC_0.relu();

        /*FC FC_1=new FC(FC_0,(float[][][][])this.d_kernel.get(4),(float[])this.d_bias.get(4));
        FC_1.relu();
        FC_1.norm();
        //FC_1.relu();*/

        OutputLayer OP=new OutputLayer(FC_0,(float[][][][])this.d_kernel.get(3),(float[])this.d_bias.get(3));
        //OP.percentify();
        OP.softmax();
        System.out.println(Arrays.toString(OP.vector));
        this.result=OP.vector;
    }


    HashMap return_d_kernel(){
        return this.d_kernel;
    }

    HashMap return_d_bias(){
        return this.d_bias;
    }

}

