package CNN;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.HashMap;

public class bpNet {

    HashMap d_bpmat=new HashMap();
    HashMap d_bpbias=new HashMap();
    HashMap d_kernel=new HashMap<Integer,float[][][][]>();
    HashMap d_bias=new HashMap<Integer,float[]>();
    String bpnet_name;
    float step;
    float countmse;

    float[] real_score;

    void clone_kernel_bias(HashMap d_kernel,HashMap d_bias){
        for(int i=0;i<d_kernel.size();i++){
            float[][][][] kernel=((float[][][][]) d_kernel.get(i));
            float[][][][] new_kernel=new float[kernel.length][kernel[0].length][kernel[0][0].length][kernel[0][0][0].length];
            for (int t=0;t<kernel.length;t++){
                for(int tt=0;tt<kernel[t].length;tt++){
                    for(int ttt=0;ttt<kernel[t][tt].length;ttt++){
                        new_kernel[t][tt][ttt]=kernel[t][tt][ttt].clone();
                    }
                }
            }
            this.d_kernel.put(i,new_kernel);
            float[] bias=((float[]) d_bias.get(i)).clone();
            this.d_bias.put(i,bias);
        }
    }

    static float[][][] txtToMat(String txtName, int bandCount, int ySize, int xSize) throws IOException {

        float[][][] mat=new float[bandCount][ySize][xSize];
        BufferedReader in = new BufferedReader(new FileReader(txtName));
        String str;
        int band=0;
        while ((str = in.readLine()) != null) {
            String[] s_i=(str.split("\\]")[0]).split("\\[")[1].split("\\, ");

            for(int y=0;y<ySize;y++){
                for(int x=0;x<xSize;x++){
                    mat[band][y][x]=(Float.parseFloat(s_i[ySize*y+x]));
                }
            }
            band+=1;
            //System.out.println(s_i[2499]);
        }

        return mat;

    }

    bpNet(String ILnamestr,HashMap d_kernel,HashMap d_bias,float[] real_score){

        //clone_kernel_bias(d_kernel,d_bias);
        this.d_kernel=d_kernel;
        this.d_bias=d_bias;
        this.real_score=real_score;


        float[][][] IL_mat=new float[0][][];

        try {
            IL_mat = txtToMat(ILnamestr,4,50,50);

        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i=0;i<IL_mat.length;i++){
            for(int ii=0;ii<IL_mat[0].length;ii++){
                for(int iii=0;iii<IL_mat[0][0].length;iii++){
                    IL_mat[i][ii][iii]/=65535;
                }
            }
        }

        bpHiddenLayer bpHL_0 = new bpHiddenLayer(IL_mat, 1, (float[][][][])this.d_kernel.get(0),(float[])this.d_bias.get(0),this.d_bpmat,this.d_bpbias);
        bpHL_0.relu();
        bpHL_0.norm();
        //bpHL_0.relu();
        bpPoolMat bpPM_0=new bpPoolMat(bpHL_0,2,"max",this.d_bpmat,this.d_bpbias);

        bpHiddenLayer bpHL_1 = new bpHiddenLayer(bpHL_0, 1, (float[][][][])this.d_kernel.get(1),(float[])this.d_bias.get(1),this.d_bpmat,this.d_bpbias);
        bpHL_1.relu();
        bpHL_1.norm();
        //bpHL_1.relu();
        bpPoolMat bpPM_1=new bpPoolMat(bpHL_1,2,"max",this.d_bpmat,this.d_bpbias);

        /*bpHiddenLayer bpHL_2 = new bpHiddenLayer(bpHL_1, 1, (float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2),this.d_bpmat,this.d_bpbias);
        bpHL_2.relu();
        bpHL_2.norm();
        //bpHL_2.relu();
        bpPoolMat bpPM_2=new bpPoolMat(bpHL_2,2,"max",this.d_bpmat,this.d_bpbias);*/

        bpFC bpFC_0=new bpFC(bpPM_1,(float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2),this.d_bpmat,this.d_bpbias);
        bpFC_0.relu();
        bpFC_0.norm();
        //bpFC_0.relu();

        /*bpFC bpFC_1=new bpFC(bpFC_0,(float[][][][])this.d_kernel.get(4),(float[])this.d_bias.get(4),this.d_bpmat,this.d_bpbias);
        bpFC_1.relu();
        bpFC_1.norm();
        //bpFC_1.relu();*/

        bpOutputLayer bpOP=new bpOutputLayer(bpFC_0,(float[][][][])this.d_kernel.get(3),(float[])this.d_bias.get(3),this.d_bpmat,this.d_bpbias);
        //bpOP.percentify();
        bpOP.softmax();
        String sstrr= Arrays.toString(this.real_score)+"last is "+Arrays.toString(bpOP.vector);
        bpFinal bf=new bpFinal(bpOP,this.real_score,(float)0.1,this.d_bpmat,this.d_bpbias,this.d_kernel,this.d_bias,"softmax");
        sstrr+="     last countmse is "+bf.countmse;
        bf.update();


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

        this.countmse = 0;
        for (int n=0;n<real_score.length;n++){
            float mse=(OP.vector[n]-real_score[n]);
            float mmse= (float) (0.5*mse*mse);

            countmse+=mmse;
        }
        countmse/=real_score.length;

        sstrr+="     now is "+Arrays.toString(OP.vector)+"     countmse:"+countmse;
        System.out.println(sstrr);

    }

    public bpNet(float[][][] IL_mat, HashMap d_kernel, HashMap d_bias, float[] real_score, float step) throws ParseException {
        long t1=System.currentTimeMillis();
        //clone_kernel_bias(d_kernel,d_bias);
        this.d_kernel=d_kernel;
        this.d_bias=d_bias;
        this.real_score=real_score;
        this.step=step;
        for(int i=0;i<IL_mat.length;i++){
            for(int ii=0;ii<IL_mat[0].length;ii++){
                for(int iii=0;iii<IL_mat[0][0].length;iii++){
                    IL_mat[i][ii][iii]/=65535;
                }
            }
        }
        //this.step=step;

        bpHiddenLayer bpHL_0 = new bpHiddenLayer(IL_mat, 1, (float[][][][])this.d_kernel.get(0),(float[])this.d_bias.get(0),this.d_bpmat,this.d_bpbias);
        bpHL_0.relu();
        bpHL_0.norm();
        //bpHL_0.relu();
        bpPoolMat bpPM_0=new bpPoolMat(bpHL_0,2,"max",this.d_bpmat,this.d_bpbias);

        bpHiddenLayer bpHL_1 = new bpHiddenLayer(bpPM_0, 1, (float[][][][])this.d_kernel.get(1),(float[])this.d_bias.get(1),this.d_bpmat,this.d_bpbias);
        bpHL_1.relu();
        bpHL_1.norm();
        //bpHL_1.relu();
        bpPoolMat bpPM_1=new bpPoolMat(bpHL_1,2,"max",this.d_bpmat,this.d_bpbias);

        /*bpHiddenLayer bpHL_2 = new bpHiddenLayer(bpHL_1, 1, (float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2),this.d_bpmat,this.d_bpbias);
        bpHL_2.relu();
        bpHL_2.norm();
        //bpHL_2.relu();
        bpPoolMat bpPM_2=new bpPoolMat(bpHL_2,2,"max",this.d_bpmat,this.d_bpbias);*/

        bpFC bpFC_0=new bpFC(bpPM_1,(float[][][][])this.d_kernel.get(2),(float[])this.d_bias.get(2),this.d_bpmat,this.d_bpbias);
        bpFC_0.relu();
        bpFC_0.norm();
        //bpFC_0.relu();

        /*bpFC bpFC_1=new bpFC(bpFC_0,(float[][][][])this.d_kernel.get(4),(float[])this.d_bias.get(4),this.d_bpmat,this.d_bpbias);
        bpFC_1.relu();
        bpFC_1.norm();
        //bpFC_1.relu();*/

        bpOutputLayer bpOP=new bpOutputLayer(bpFC_0,(float[][][][])this.d_kernel.get(3),(float[])this.d_bias.get(3),this.d_bpmat,this.d_bpbias);
        //bpOP.percentify();
        bpOP.softmax();
        String sstrr=Arrays.toString(this.real_score)+"last is "+Arrays.toString(bpOP.vector);
        bpFinal bf=new bpFinal(bpOP,this.real_score,this.step,this.d_bpmat,this.d_bpbias,this.d_kernel,this.d_bias,"softmax");
        //sstrr+="     last countmse is "+bf.countmse;
        bf.update();


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

        this.countmse = 0;
        for (int n=0;n<real_score.length;n++){
            float mse=(OP.vector[n]-real_score[n]);
            float mmse= (float) (0.5*mse*mse);

            countmse+=mmse;
        }
        countmse/=real_score.length;

        long t2=System.currentTimeMillis();

        SimpleDateFormat dateformat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String dateStr = dateformat.format(System.currentTimeMillis());

        sstrr+="  now is "+Arrays.toString(OP.vector)+"  current_time:"+dateStr+" usetime="+(t2-t1) +"ms";
        //sstrr+="     now is "+Arrays.toString(OP.vector)+"     countmse:"+countmse;
        System.out.println(sstrr);
    }


}