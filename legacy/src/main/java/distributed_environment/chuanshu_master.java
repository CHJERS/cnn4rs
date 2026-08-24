package distributed_environment;

import CNN.Net;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;



public class chuanshu_master{
    //private String slave_IP;
    //private int slave_port;
    //private String trans_json;

    static HashMap d_kernel=new HashMap<Integer,float[][][][]>();
    static HashMap d_bias=new HashMap<Integer,float[]>();

    /*chuanshu_master(String slave_IP,int slave_port,String str_json){
        this.slave_IP=slave_IP;
        this.slave_port=slave_port;
        this.trans_json=str_json;
    }

    static void clone_kernel_bias(HashMap d_kernel, HashMap d_bias){

        for(int i = 0; i< chuanshu_master.d_kernel.size(); i++){
            float[][][][] kernel=((float[][][][]) chuanshu_master.d_kernel.get(i));
            float[][][][] new_kernel=new float[kernel.length][kernel[0].length][kernel[0][0].length][kernel[0][0][0].length];
            for (int t=0;t<kernel.length;t++){
                for(int tt=0;tt<kernel[t].length;tt++){
                    for(int ttt=0;ttt<kernel[t][tt].length;ttt++){
                        new_kernel[t][tt][ttt]=kernel[t][tt][ttt].clone();
                    }
                }
            }
            d_kernel.put(i,new_kernel);
            float[] bias=((float[]) chuanshu_master.d_bias.get(i)).clone();
            d_bias.put(i,bias);
        }

    }*/

    static void update_kernel_bias(HashMap hm_d){
        System.out.println("it is doing update");
        HashMap new_d_kernel=new HashMap<Integer,float[][][][]>();
        HashMap new_d_bias=new HashMap<Integer,float[]>();
        for(int i = 0; i< chuanshu_master.d_kernel.size(); i++){
            float[][][][] kernel=((float[][][][]) chuanshu_master.d_kernel.get(i));
            float[][][][] new_kernel=new float[kernel.length][kernel[0].length][kernel[0][0].length][kernel[0][0][0].length];
            float[] bias=new float[kernel.length];
            new_d_kernel.put(i,new_kernel);
            new_d_bias.put(i,bias);
        }

        Iterator iter = hm_d.entrySet().iterator();
        boolean hm_tf=iter.hasNext();

        int sbn=0;

        while (hm_tf) {
            sbn+=1;
            Map.Entry entry = (Map.Entry) iter.next();
            HashMap kernel_bias = (HashMap)entry.getValue();
            HashMap d_kernel=(HashMap)kernel_bias.get("kernel");
            HashMap d_bias=(HashMap)kernel_bias.get("bias");
            int hm_d_size=hm_d.size();
            hm_tf=iter.hasNext();

            if(hm_tf){
                for(int nu=0;nu<d_kernel.size();nu++){
                    float[][][][] last_kernel=(float[][][][])d_kernel.get(nu);
                    float[][][][] new_kernel=(float[][][][])new_d_kernel.get(nu);

                    for (int t=0;t<last_kernel.length;t++){
                        for(int tt=0;tt<last_kernel[t].length;tt++){
                            for(int ttt=0;ttt<last_kernel[t][tt].length;ttt++){
                                for(int tttt=0;tttt<last_kernel[t][tt][ttt].length;tttt++){
                                    new_kernel[t][tt][ttt][tttt]+=last_kernel[t][tt][ttt][tttt];
                                }
                            }
                        }
                        ((float[]) new_d_bias.get(nu))[t]+=((float[])d_bias.get(nu))[t];
                    }
                }
            }
            else{

                for(int nu=0;nu<d_kernel.size();nu++){
                    float[][][][] last_kernel=(float[][][][])d_kernel.get(nu);
                    float[][][][] new_kernel=(float[][][][])new_d_kernel.get(nu);

                    for (int t=0;t<last_kernel.length;t++){
                        for(int tt=0;tt<last_kernel[t].length;tt++){
                            for(int ttt=0;ttt<last_kernel[t][tt].length;ttt++){
                                for(int tttt=0;tttt<last_kernel[t][tt][ttt].length;tttt++){
                                    new_kernel[t][tt][ttt][tttt]+=last_kernel[t][tt][ttt][tttt];
                                    new_kernel[t][tt][ttt][tttt]/=hm_d_size;
                                }
                            }
                        }
                        ((float[]) new_d_bias.get(nu))[t]+=((float[])d_bias.get(nu))[t];
                        ((float[]) new_d_bias.get(nu))[t]/=hm_d_size;
                    }
                }
            }

            /*String kernel_file="H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//train//kernel_bias//kernel//kernel"+nnn+".txt";
            nnn+=1;

            try {
                write_txt1(d_kernel,kernel_file);
            } catch (IOException e) {
                e.printStackTrace();
            }*/


        }
        chuanshu_master.d_kernel=new_d_kernel;
        chuanshu_master.d_bias=new_d_bias;

        System.out.println("updating is done");

    }

    static void import_kernel_bias(String kernel_file, String bias_file) throws IOException {
        BufferedReader in1 = new BufferedReader(new FileReader(kernel_file));
        BufferedReader in2 = new BufferedReader(new FileReader(bias_file));

        HashMap new_d_kernel=new HashMap<Integer,float[][][][]>();
        HashMap new_d_bias=new HashMap<Integer,float[]>();
        String str1;
        String str2;
        int nn=0;
        float[][][][] kernel;
        String[] kernel_0;
        String[] kernel_1;
        String[] kernel_2;
        String[] kernel_3;
        float[] bias;
        String[] bias_0;

        while ((str1 = in1.readLine()) != null) {

            kernel_0=((((str1.split("\\]\\]\\]\\]"))[0]).split("\\[\\[\\[\\["))[1]).split("\\]\\]\\], \\[\\[\\[");

            kernel_1=kernel_0[0].split("\\]\\], \\[\\[");

            kernel_2=kernel_1[0].split("\\], \\[");

            kernel_3=kernel_2[0].split("\\, ");

            kernel=new float[kernel_0.length][kernel_1.length][kernel_2.length][kernel_3.length];

            for(int t=0;t<kernel_0.length;t++){
                kernel_1=kernel_0[t].split("\\]\\], \\[\\[");
                for(int tt=0;tt<kernel_1.length;tt++){
                    kernel_2=kernel_1[tt].split("\\], \\[");
                    for(int ttt=0;ttt<kernel_2.length;ttt++){
                        kernel_3=kernel_2[ttt].split("\\, ");
                        for(int tttt=0;tttt<kernel_3.length;tttt++){
                            kernel[t][tt][ttt][tttt]=Float.parseFloat(kernel_3[tttt]);
                        }
                    }
                }
            }

            str2 = in2.readLine();
            bias_0=(str2.split("\\]")[0]).split("\\[")[1].split("\\, ");
            bias=new float[bias_0.length];
            for(int t=0;t<bias_0.length;t++){
                bias[t]=Float.parseFloat(bias_0[t]);
            }

            new_d_kernel.put(nn,kernel);
            new_d_bias.put(nn,bias);
            nn+=1;
        }
        chuanshu_master.d_kernel=new_d_kernel;
        chuanshu_master.d_bias=new_d_bias;
    }



    static void write_txt(int iter,int n,String dirname) throws IOException {
        String kernel_file=dirname+"//kernel//kernel_"+iter+"_"+n+".txt";
        String bias_file=dirname+"//bias//bias_"+iter+"_"+n+".txt";
        BufferedWriter out1 = new BufferedWriter(new FileWriter(kernel_file));
        BufferedWriter out2 = new BufferedWriter(new FileWriter(bias_file));
        for (int i = 0; i < chuanshu_master.d_kernel.size(); i++) {
            out1.write(Arrays.deepToString((float[][][][]) chuanshu_master.d_kernel.get(i)));
            out1.write("\n");
            out2.write(Arrays.toString((float[])(chuanshu_master.d_bias.get(i))));
            out2.write("\n");
        }
        out1.close();
        out2.close();
    }

    static float[][][] readMatFromTxt(String mat_file, int bandCount, int ySize, int xSize) throws IOException {

        float[][][] mat=new float[bandCount][ySize][xSize];
        BufferedReader read_mat = new BufferedReader(new FileReader(mat_file));
        String str1;
        String str_mat="";
        while ((str1 = read_mat.readLine()) != null) {
            str_mat+=str1;
            str_mat+="\n";
        }

        read_mat.close();
        String[] s_mat=str_mat.split("\\]");
        int band=0;
        for(int i=0;i<bandCount;i++){
            String[] s_i=(s_mat[i].split("\\[")[1]).split("\\, ");

            for(int y=0;y<50;y++){
                for(int x=0;x<50;x++){
                    mat[band][y][x]=Float.parseFloat(s_i[50*y+x]);
                }
            }
            band+=1;
        }

        return mat;
    }

    static void write_kb_ser(int iter,int n,String dirname)throws IOException{
        String kernel_file=dirname+"//kernel//kernel_"+iter+"_"+n+".ser";
        String bias_file=dirname+"//bias//bias_"+iter+"_"+n+".ser";

        ObjectOutputStream kernel_out = new ObjectOutputStream(new FileOutputStream(kernel_file));
        kernel_out.writeObject(chuanshu_master.d_kernel);

        ObjectOutputStream bias_out = new ObjectOutputStream(new FileOutputStream(bias_file));
        bias_out.writeObject(chuanshu_master.d_bias);

        kernel_out.close();
        bias_out.close();

    }

    static float[][][] readMatFromSer(String filename) throws IOException, ClassNotFoundException {
        float[][][] mat;
        ObjectInputStream in= new ObjectInputStream(new FileInputStream(filename));
        mat=(float[][][])in.readObject();
        in.close();
        return mat;
    }

    static void importKBFromSer(String dirname, int iter, int loop)throws IOException, ClassNotFoundException{
        String kernel_file=dirname+"//kernel//kernel_"+iter+"_"+loop+".ser";
        String bias_file=dirname+"//bias//bias_"+iter+"_"+loop+".ser";
        ObjectInputStream in= new ObjectInputStream(new FileInputStream(kernel_file));
        chuanshu_master.d_kernel=(HashMap) in.readObject();
        in.close();
        in= new ObjectInputStream(new FileInputStream(bias_file));
        chuanshu_master.d_bias=(HashMap) in.readObject();
        in.close();

    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {

/*        String mat_url="C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//S//fangkuai_1795_688.ser";

        float[][][] ori_mat=readMatFromSer(mat_url);

        int band=ori_mat.length;
        int mat_y=ori_mat[0].length;
        int mat_x=ori_mat[0][0].length;

        float[][][] new_net_mat=new float[ori_mat.length][mat_y][mat_x];
        for (int i=0;i<band;i++){
            for (int ii=0;ii<mat_y;ii++){
                new_net_mat[i][ii]=ori_mat[i][ii].clone();
            }
        }

        Net n=new Net(new_net_mat);
        chuanshu_master.d_kernel=n.return_d_kernel();
        chuanshu_master.d_bias=n.return_d_bias();

        //importKBFromSer("C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//KB//",1,12);
        //Net n=new Net(new_net_mat,chuanshu_master.d_kernel,chuanshu_master.d_bias);

        String dir_st_ser="C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//S//";
        File f1 = new File(dir_st_ser);
        String[] s1 = f1.list();

        String dir_ld_ser="C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//L//";
        File f2 = new File(dir_ld_ser);
        String[] s2 = f2.list();


        int count_threads=16;
        int half_ct = count_threads / 2;
        int iter_count=100;
        float step=(float)0.01;
        HashMap h1 ;
        Random random=new Random();
        int sj;
        for (int iter=0;iter<iter_count;iter++){
            int loopcount=s1.length/count_threads;
            for(int loop=0;loop<loopcount;loop++) {
                long t=System.currentTimeMillis();

                HashMap hm_d = new HashMap();

                HashMap loc_mat = new HashMap();


                for (int i = 0; i < count_threads; i++) {
                    HashMap kernel_bias = new HashMap();
                    hm_d.put(i, kernel_bias);

                    h1 = new HashMap();
                    sj=random.nextInt(496);
                    int zl=random.nextInt(2);
                    if (zl==1){
                        h1.put("mat_url", dir_st_ser + s1[sj]);
                        h1.put("leixing", new float[]{1, 0});

                    }
                    else{
                        h1.put("mat_url", dir_ld_ser + s2[sj]);
                        h1.put("leixing", new float[]{0, 1});
                    }
                    h1.put("step", step);
                    loc_mat.put(i, h1);

                }

                HashMap t_control = new HashMap<Integer, Thread>();

                int ac = 10;

                for (int c = 0; c < ac; c++) {
                    HashMap lh = (HashMap) loc_mat.get(c);
                    t_control.put(c, new Thread(new master_thread("192.168.50.113", 8888 + c, (HashMap) hm_d.get(c), (String) lh.get("mat_url"), (float[]) lh.get("leixing"), (float)lh.get("step"))));
                    ((Thread) t_control.get(c)).start();
                }

                int ic = 6;

                for (int c = 0; c < ic; c++) {
                    HashMap lh = (HashMap) loc_mat.get(c + ac);
                    t_control.put(c + ac, new Thread(new master_thread("192.168.50.213", 8888 + c, (HashMap) hm_d.get(c + ac), (String) lh.get("mat_url"), (float[]) lh.get("leixing"), (float)lh.get("step"))));
                    ((Thread) t_control.get(c + ac)).start();
                }

                for (int c = 0; c < count_threads; c++) {
                    ((Thread) t_control.get(c)).join();
                }

                update_kernel_bias(hm_d);

                new_net_mat = new float[ori_mat.length][mat_y][mat_x];                  //注释掉
                for (int i = 0; i < band; i++) {
                    for (int ii = 0; ii < mat_y; ii++) {
                        new_net_mat[i][ii] = ori_mat[i][ii].clone();
                    }
                }

                new Net(new_net_mat, chuanshu_master.d_kernel, chuanshu_master.d_bias);         //注释掉



                write_kb_ser(iter,loop,"C://Users//CHJERS_1//Desktop//学术学位硕士论文格式及相关表格//KB//NIR_SL//KB//");


                testAccuracy.testAcc(chuanshu_master.d_kernel,chuanshu_master.d_bias);
                String qbs="using ";
                qbs+=(System.currentTimeMillis()-t);
                System.out.println(qbs);
                String end=iter+"_"+loop+"is done";
                System.out.println(end);
            }

        }*/



        String mat_url="C://Temp//zb.ser";
        float[][][] ori_mat=readMatFromSer(mat_url);

        //Net n=new Net(ori_mat);
        //chuanshu_master.d_kernel=n.return_d_kernel();
        //chuanshu_master.d_bias=n.return_d_bias();
        //write_kb_ser(0,0,"C://Temp//KB//");

        importKBFromSer("C://Temp//KB//",0,0);

        int band=ori_mat.length;
        int mat_y=ori_mat[0].length;
        int mat_x=ori_mat[0][0].length;


        int tz=16;
        int zg=64;

        HashMap hm_d;

        long t=System.currentTimeMillis();
        int iterc=(int)Math.ceil(zg/tz);
        for (int iter=0;iter<iterc;iter++) {



            long t1=System.currentTimeMillis();
            hm_d = new HashMap<Integer, HashMap>(tz);
            HashMap t_control = new HashMap<Integer, Thread>();

            for (int tcount = 0; tcount < tz; tcount++) {
                HashMap kernel_bias1 = new HashMap();
                hm_d.put(tcount, kernel_bias1);
            }

            int ac = 14;

            for (int c = 0; c < ac; c++) {
                t_control.put(c, new Thread(new master_thread("192.168.50.113", 8888 + c, (HashMap) hm_d.get(c), mat_url, new float[]{1, 0},(float)0.1)));
                ((Thread) t_control.get(c)).start();
            }


            int ic = 2;
            for (int c = 0; c < ic; c++) {
                t_control.put(c+ac, new Thread(new master_thread("192.168.50.213", 8888 + c, (HashMap) hm_d.get(c + ac), mat_url, new float[]{1, 0},(float)0.1)));
                ((Thread) t_control.get(c + ac)).start();
            }


            for (int c = 0; c < ic+ac; c++) {
                ((Thread) t_control.get(c)).join();
                //((Thread)t_control.get(c)).stop();
            }

            update_kernel_bias(hm_d);


            float[][][] new_net_mat=new float[band][mat_y][mat_x];
            for (int i=0;i<band;i++){
                for (int ii=0;ii<mat_y;ii++){
                    new_net_mat[i][ii]=ori_mat[i][ii].clone();
                }
            }

            new Net(new_net_mat, chuanshu_master.d_kernel, chuanshu_master.d_bias);
            long t2=System.currentTimeMillis();

            SimpleDateFormat dateformat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            String dateStr = dateformat.format(System.currentTimeMillis());

            String strr="current_time:"+dateStr+"   usetime="+(t2-t1)+" ms";
            System.out.println(strr);
        }
        long zsj=System.currentTimeMillis()-t;
        String zsjs="toatal uses "+zsj+"ms";
        System.out.println(zsjs);



    }
}