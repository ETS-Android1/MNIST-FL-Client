package com.android.example.inferenceapp;

import android.os.Environment;

import com.jjoe64.graphview.series.DataPoint;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Model {
    private static final int MNIST_LEN = 10;
    private final String mPath, mName;
    private Map<Integer, int[]> mLabelData;
    private double mAvgAcc;

    public Model(String name){
        this.mName = name;
        this.mLabelData = new HashMap<>();
        String PATH = Environment.getExternalStorageDirectory().getPath()+"/MNIST_FL/Models";
        this.mPath = PATH +"/"+ name;
        this.getCountData();
    }

    public String getmName(){
        return this.mName;
    }

    public String getmPath() {
        return mPath;
    }

    public double getAvgAccuracy(){
        getAccuracy();
        return mAvgAcc/(MNIST_LEN*10.0);
    }

    public DataPoint[] getAccuracy(){
        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];
        Arrays.fill(dataPoints, new DataPoint(0, 0));
        if(this.mLabelData.isEmpty()) return dataPoints;
        for(int i=0; i<MNIST_LEN; i++){
            int totalCount = 0;
            for(int count : Objects.requireNonNull(this.mLabelData.get(i))){
                totalCount += count;
            }
            dataPoints[i] = new DataPoint(i, (totalCount==0)?0:(Objects.requireNonNull(this.mLabelData.get(i))[i]/(totalCount*1.0))*100);
            mAvgAcc += dataPoints[i].getY();
        }
        return dataPoints;
    }

    public DataPoint[] getF1Score(){
        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];
        Arrays.fill(dataPoints, new DataPoint(0, 0));
        if(this.mLabelData.isEmpty()) return dataPoints;
        int[][] data = new int[MNIST_LEN][MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            data[i] = this.mLabelData.get(i).clone();
        }
        int[] precision = new int[MNIST_LEN];
        int[] recall = new int[MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            for(int j=0; j<MNIST_LEN; j++){
                precision[i] += data[i][j];
            }
        }
        for(int j=0; j<MNIST_LEN; j++){
            for (int i = 0; i < MNIST_LEN; i++){
                recall[j] += data[i][j];
            }
        }

//        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            double p = precision[i]==0?0:data[i][i]/(precision[i]*1.0);
            double r = recall[i]==0?0:data[i][i]/(recall[i]*1.0);
            double f1score = (p+r)==0?0:(2*p*r)/(p+r);
            dataPoints[i] = new DataPoint(i, f1score*100);
        }
        return dataPoints;
    }

    public void incrementCount(int prediction, int label){
        if(this.mLabelData.isEmpty()){
            this.mLabelData = new HashMap<>();
            for(int i=0; i<10; i++){
                this.mLabelData.put(i, new int[MNIST_LEN]);
            }
        }
        if(this.mLabelData.containsKey(prediction)) {
            Objects.requireNonNull(this.mLabelData.get(label))[prediction]++;
        }
        for(int i=0; i<10; i++){
            System.out.println("count "+i+" "+ Arrays.toString(this.mLabelData.get(i)));
        }
    }

    public void setCountData(){
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(mPath+"/labelCount.txt");
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
            objectOutputStream.writeObject(this.mLabelData);
            objectOutputStream.close();
            fileOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void getCountData(){
        try {
            FileInputStream fileInputStream = new FileInputStream(this.mPath+"/labelCount.txt");
            ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
            this.mLabelData = (HashMap)objectInputStream.readObject();
            objectInputStream.close();
            fileInputStream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public boolean equals(Object obj) {
        if(obj==null) return false;
        return this.mPath.equals(((Model)(obj)).mPath);
    }

    @Override
    public int hashCode() {
        return this.mPath.hashCode();
    }
}
