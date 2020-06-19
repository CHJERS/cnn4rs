package distributed_environment;


import java.io.*;
import java.net.Socket;
import java.util.HashMap;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;


class master_thread implements Runnable{

    private String IP;
    private int port;
    private HashMap kernel_bias;
    //private String zhibei_mat;
    private float[] realscore;
    private float[][][] mat;
    private float step;
    //private String str_json;

    master_thread(String IP,int port,HashMap kernel_bias,String zhibei_mat,float[] realscore,float step) throws IOException, ClassNotFoundException {
        this.IP=IP;
        this.port=port;
        this.kernel_bias=kernel_bias;

        ObjectInputStream in = new ObjectInputStream(new FileInputStream(zhibei_mat));
        this.mat = (float[][][]) in.readObject();
        in.close();

        this.realscore=realscore;
        this.step=step;
    }

    void importMatToJson(JSONObject json,String matfile) throws IOException {
        BufferedReader read_mat = new BufferedReader(new FileReader(matfile));
        String str1;
        String str_mat="";
        while ((str1 = read_mat.readLine()) != null) {
            str_mat+=str1;
            str_mat+="\n";
        }
        json.put("IL_mat",str_mat);
        read_mat.close();
    }

    private void jsonarrayToMap(HashMap td_kernel,HashMap td_bias) throws ArrayIndexOutOfBoundsException{

        //master_thread.write_je(td_kernel.toString());

        HashMap new_d_kernel=new HashMap<Integer,float[][][][]>();
        HashMap new_d_bias=new HashMap<Integer,float[]>();
        String str1;
        String str2;
        float[][][][] kernel;
        String[] kernel_0;
        String[] kernel_1;
        String[] kernel_2;
        String[] kernel_3;
        float[] bias;
        String[] bias_0;
        for(int i=0;i<td_kernel.size();i++){
            //String si=Integer.toString(i);
            str1=td_kernel.get(i).toString();

            kernel_0=((((str1.split("\\]\\]\\]\\]"))[0]).split("\\[\\[\\[\\["))[1]).split("\\]\\]\\],\\[\\[\\[");

            kernel_1=kernel_0[0].split("\\]\\],\\[\\[");

            kernel_2=kernel_1[0].split("\\],\\[");

            kernel_3=kernel_2[0].split("\\,");

            kernel=new float[kernel_0.length][kernel_1.length][kernel_2.length][kernel_3.length];

            for(int t=0;t<kernel_0.length;t++){
                kernel_1=kernel_0[t].split("\\]\\],\\[\\[");
                for(int tt=0;tt<kernel_1.length;tt++){
                    kernel_2=kernel_1[tt].split("\\],\\[");
                    for(int ttt=0;ttt<kernel_2.length;ttt++){
                        kernel_3=kernel_2[ttt].split("\\,");
                        for(int tttt=0;tttt<kernel_3.length;tttt++){
                            kernel[t][tt][ttt][tttt]=Float.parseFloat(kernel_3[tttt]);
                        }
                    }
                }
            }

            str2 = td_bias.get(i).toString();
            bias_0=(str2.split("\\]")[0]).split("\\[")[1].split("\\,");
            bias=new float[bias_0.length];
            for(int t=0;t<bias_0.length;t++){
                bias[t]=Float.parseFloat(bias_0[t]);
            }

            //this.d_kernel.put(i,kernel);
            //this.d_bias.put(i,bias);
            new_d_kernel.put(i,kernel);
            new_d_bias.put(i,bias);
        }
        //this.d_kernel=new_d_kernel;
        //this.d_bias=new_d_bias;
        this.kernel_bias.put("kernel",new_d_kernel);
        this.kernel_bias.put("bias",new_d_bias);
    }

    static String MatFromTxtToString(String matfile) throws IOException{
        BufferedReader read_mat = new BufferedReader(new FileReader(matfile));
        String str1;
        String str_mat="";
        while ((str1 = read_mat.readLine()) != null) {
            str_mat+=str1;
            str_mat+="\n";
        }
        read_mat.close();
        return str_mat;
    }

    public void run() {

        //System.out.println("tongxinqidong");

        try {



            HashMap fs_hm=new HashMap();
            fs_hm.put("realscore", this.realscore);
            fs_hm.put("d_kernel",chuanshu_master.d_kernel);
            fs_hm.put("d_bias",chuanshu_master.d_bias);
            //String str_mat=this.MatFromTxtToString(this.zhibei_mat);
            fs_hm.put("IL_mat",this.mat);
            fs_hm.put("step",this.step);

            Socket socket = new Socket(this.IP, this.port);
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(socket.getOutputStream());
            objectOutputStream.writeObject(fs_hm);
            objectOutputStream.flush();
            ObjectInputStream objectInputStream = new ObjectInputStream(socket.getInputStream());
            HashMap js_hm=(HashMap) objectInputStream.readObject();
            socket.close();

            this.kernel_bias.put("kernel",(HashMap)js_hm.get("kernel"));
            this.kernel_bias.put("bias",(HashMap)js_hm.get("bias"));

            /*JSONObject chuanshu_json = new JSONObject();

            String str_kernel="";
            String str_bias="";

            for(int i = 0; i< chuanshu_master.d_kernel.size(); i++){
                str_kernel+=(Arrays.deepToString((float[][][][]) chuanshu_master.d_kernel.get(i))+"\n");
                str_bias+=(Arrays.toString((float[]) chuanshu_master.d_bias.get(i))+"\n");
            }

            chuanshu_json.put("d_kernel",str_kernel);
            chuanshu_json.put("d_bias",str_bias);

            try {
                this.importMatToJson(chuanshu_json,this.zhibei_mat);
            } catch (IOException e) {
                e.printStackTrace();
            }

            chuanshu_json.put("realscore", this.realscore);
            chuanshu_json.put("leixing","zhibei");

            String str_json=JSON.toJSONString(chuanshu_json);
            Socket socket = new Socket(this.IP, this.port);
            BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
            bufferedWriter.write(str_json);
            bufferedWriter.write("\n");
            bufferedWriter.flush();
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream(), "UTF-8"));
            String slave_jstr= bufferedReader.readLine();
            socket.close();
            master_thread.write_je(slave_jstr);
            JSONObject slave_json = JSONObject.parseObject(slave_jstr);
            HashMap td_kernel= JSON.parseObject(slave_json.getString("d_kernel"),HashMap.class);
            HashMap td_bias= JSON.parseObject(slave_json.getString("d_bias"),HashMap.class);
            master_thread.write_jek(td_kernel.toString());
            master_thread.write_jeb(td_bias.toString());
            jsonarrayToMap(td_kernel,td_bias);


            /*try{
                JSONObject slave_json = JSONObject.parseObject(slave_jstr);
                HashMap td_kernel= JSON.parseObject(slave_json.getString("d_kernel"),HashMap.class);
                HashMap td_bias= JSON.parseObject(slave_json.getString("d_bias"),HashMap.class);
                try{
                    jsonarrayToMap(td_kernel,td_bias);}
                catch(ArrayIndexOutOfBoundsException ae){
                    System.out.println(td_kernel.get(0));
                    System.out.println(td_bias.get(0));
                }
            }
            catch(com.alibaba.fastjson.JSONException je){
                write_je(slave_jstr);
            }*/


        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    static void write_je(String slave_jstr){
        String fn="H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//weifen//txt//Binary//je//je_"+System.currentTimeMillis()+"_"+Thread.currentThread().getId()+".txt";
        BufferedWriter out1 = null;
        try {
            out1 = new BufferedWriter(new FileWriter(fn));
            out1.write(slave_jstr);
            out1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    static void write_jek(String skernel){
        String fn="H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//weifen//txt//Binary//je_kernel//je_"+System.currentTimeMillis()+"_"+Thread.currentThread().getId()+".txt";
        BufferedWriter out1 = null;
        try {
            out1 = new BufferedWriter(new FileWriter(fn));
            out1.write(skernel);
            out1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void write_jeb(String sbias){
        String fn="H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//weifen//txt//Binary//je_bias//je_"+System.currentTimeMillis()+"_"+Thread.currentThread().getId()+".txt";
        BufferedWriter out1 = null;
        try {
            out1 = new BufferedWriter(new FileWriter(fn));
            out1.write(sbias);
            out1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
