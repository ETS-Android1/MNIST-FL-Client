package com.android.example.inferenceapp;

import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.icu.text.SimpleDateFormat;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.text.InputType;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.Viewport;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.DataPointInterface;
import com.jjoe64.graphview.series.LineGraphSeries;
import com.jjoe64.graphview.series.OnDataPointTapListener;
import com.jjoe64.graphview.series.Series;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Objects;
import java.util.UUID;

import static android.Manifest.permission.INTERNET;
import static android.Manifest.permission.MANAGE_EXTERNAL_STORAGE;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class MainActivity extends AppCompatActivity {
    private CanvasView canvasView;
    private EditText input;
    private Module module_layer_1;
    private static Module module_layer_2;
    private AlertDialog.Builder builder;
    private static ProgressDialog progressDialog;
    private static Context context;

    private static String uniqueID = null;
    private static final String PREF_UNIQUE_ID = "PREF_UNIQUE_ID";
    private static final String TOTAL_COUNT = "TOTAL_COUNT";
    private static final String TOTAL_CORRECT = "TOTAL_CORRECT";

    private static final int PERMISSION_REQUEST_CODE = 200;
    private static final int MNIST_LEN = 10;

    private String label = "";
    private static HashMap<Integer, int[]> labelData;

    private static final String PATH = Environment.getExternalStorageDirectory().getPath();
    private static final String dir = PATH+"/MNIST_FL";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        canvasView = findViewById(R.id.main_canvas);
        Button predict = findViewById(R.id.predict);
        Button update = findViewById(R.id.update_model);
        context = getApplicationContext();
        String date = new SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(new Date());
        builder = new AlertDialog.Builder(this);
        input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);

        if(!checkPermission()){
            requestPermission();
        }
        File directory = new File(dir);
        if(!directory.exists()){
            directory.mkdir();
        }
        File models = new File(dir+"/Models");
        File images = new File(dir+"/images");
        if(!models.exists()){
            models.mkdir();
        }
        if(!images.exists()){
            images.mkdir();
        }

        progressDialog = new ProgressDialog(MainActivity.this);
        progressDialog.setMessage("Model updating...");
        progressDialog.setIndeterminate(false);
        progressDialog.setMax(100);
        progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        try {
            module_layer_1 = Module.load(assetFilePath(getApplicationContext(), "model_mnist_cnn_1.pt"));
            module_layer_2 = Module.load(assetFilePath(getApplicationContext(), "model_mnist_cnn_2.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        labelData = new HashMap<>();
        for(int i=0; i<10; i++){
            labelData.put(i, new int[MNIST_LEN]);
        }

        update.setOnClickListener(view -> update());

        predict.setOnClickListener(view -> {
            try {
                recognize();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.client_menu, menu);
        return true;
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch(item.getItemId()){
            case R.id.update:
                update();
                return true;
            case R.id.graph:
                getAccuracy();
                return true;
            case R.id.f1_score:
                getF1Score();
                return true;
            case R.id.bugs:
                Toast.makeText(this, "You are being redirected to a browser...", Toast.LENGTH_SHORT).show();
                String issuesUrl = "https://github.com/nagendar-pm/MNIST-FL-Client/issues";
                Intent intent = new Intent(Intent.ACTION_VIEW);
                intent.setData(Uri.parse(issuesUrl));
                startActivity(intent);
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private boolean checkPermission() {
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int result2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        int result4 = ContextCompat.checkSelfPermission(getApplicationContext(), INTERNET);
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            int result3 = ContextCompat.checkSelfPermission(getApplicationContext(), MANAGE_EXTERNAL_STORAGE);
            return result1== PackageManager.PERMISSION_GRANTED && result2== PackageManager.PERMISSION_GRANTED
                   && result3==PackageManager.PERMISSION_GRANTED && result4== PackageManager.PERMISSION_GRANTED;
        }
        return result1== PackageManager.PERMISSION_GRANTED && result2== PackageManager.PERMISSION_GRANTED
                && result4== PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET, MANAGE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        }
        else{
            ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET}, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0) {
                boolean writeAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                boolean readAccepted = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                boolean internetAccepted = grantResults[2] == PackageManager.PERMISSION_GRANTED;
                boolean isR = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R;
                boolean manageAccepted = false;
                if (isR)
                    manageAccepted = grantResults[3] == PackageManager.PERMISSION_GRANTED;
                if (writeAccepted && readAccepted && internetAccepted && isR == manageAccepted)
                    Toast.makeText(this, "Permission Granted, Now app can access storage", Toast.LENGTH_SHORT).show();
                else {
                    System.out.println(writeAccepted + " " + readAccepted + " " + internetAccepted + " " + isR + " " + manageAccepted);
                    Toast.makeText(this, "Permission Denied, App cannot access storage", Toast.LENGTH_SHORT).show();

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        if (shouldShowRequestPermissionRationale(MANAGE_EXTERNAL_STORAGE)) {
                            showMessageOKCancel("You need to allow access to all the permissions",
                                    (dialog, which) -> requestPermissions(new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET, MANAGE_EXTERNAL_STORAGE},
                                            PERMISSION_REQUEST_CODE));
                        }
                    } else {
                        if (shouldShowRequestPermissionRationale(WRITE_EXTERNAL_STORAGE)) {
                            showMessageOKCancel("You need to allow access to all the permissions",
                                    (dialog, which) -> requestPermissions(new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET},
                                            PERMISSION_REQUEST_CODE));
                        }
                    }
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private void showMessageOKCancel(String message, DialogInterface.OnClickListener okListener) {
        new AlertDialog.Builder(MainActivity.this)
                .setMessage(message)
                .setPositiveButton("OK", okListener)
                .setNegativeButton("Cancel", null)
                .create()
                .show();
    }

    public void clearCanvas(View view) {
        canvasView.clearCanvas();
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private static void post(String json) throws Exception{
        new Thread(() -> {
            try {
                String charset = "UTF-8";
                HttpURLConnection connection = (HttpURLConnection) new URL("https://fl-mnist-server.herokuapp.com/train").openConnection();
                connection.setDoOutput(true); // Triggers POST.
                connection.setRequestProperty("Accept-Charset", charset);
                connection.setRequestProperty("Content-Type", "application/json;charset=" + charset);

                try (OutputStream output = connection.getOutputStream()) {
                    output.write(json.getBytes(charset));
                }
                InputStream response = connection.getInputStream();
//                    System.out.println(response.toString());
            }
            catch (IOException e) {
                System.out.println("exception "+e);
                e.printStackTrace();
            }
        }).start();
    }

    private void update(){
        UpdateModule updateModule = new UpdateModule();
        updateModule.execute("https://fl-mnist-server.herokuapp.com/update");
    }

    private void recognize() throws Exception {
        canvasView.screenShot(dir);
        Bitmap bitmap = BitmapFactory.decodeFile(dir+"/img.jpg");
        FloatBuffer inputBuff = Tensor.allocateFloatBuffer(bitmap.getHeight() * bitmap.getWidth());

        final double GS_RED = 0.299;
        final double GS_GREEN = 0.587;
        final double GS_BLUE = 0.114;

        for(int y=0; y<bitmap.getHeight(); y++){
            for(int x=0; x<bitmap.getWidth(); x++){
                int pixel = bitmap.getPixel(x, y);
                int R = Color.red(pixel);
                int G = Color.green(pixel);
                int B = Color.blue(pixel);
                double val = 255-(R * GS_RED + G * GS_GREEN + B * GS_BLUE);
                val /= 255;
                inputBuff.put((float) val);
            }
        }

        Tensor inputTensor = Tensor.fromBlob(inputBuff, new long[]{1, 1, bitmap.getHeight(), bitmap.getWidth()});

//        For Debugging input
//        float[] toPrint = inputTensor.getDataAsFloatArray();
//        for(int i=0; i<bitmap.getHeight()*bitmap.getWidth(); i+=bitmap.getWidth()){
//            Log.d("input ", Arrays.toString(Arrays.copyOfRange(toPrint, i, i+28)));
//        }

        IValue out = module_layer_1.forward(IValue.from(inputTensor));
        final Tensor outTensor = out.toTensor();
        IValue out2 = module_layer_2.forward(IValue.from(outTensor));

        final IValue[] outputTuple = out2.toTuple();
        final float[] outputPrediction = outputTuple[0].toTensor().getDataAsFloatArray();

        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < outputPrediction.length; i++) {
            if (outputPrediction[i] > maxScore) {
                maxScore = outputPrediction[i];
                maxScoreIdx = i;
            }
        }

        if(input.getParent()!=null)
            ((ViewGroup)input.getParent()).removeView(input);
        int finalMaxScoreIdx = maxScoreIdx;
        builder.setMessage("Predicted as " + maxScoreIdx + "\nPlease provide the original Label!")
            .setView(input)
            .setCancelable(false)
            .setPositiveButton("Submit", (dialog, id) -> {
                label = input.getText().toString();
                if(!label.trim().equals("") && label.length()==1 && label.charAt(0)-'0'>=0 && label.charAt(0)-'0'<=9) {
                    input.setText("");
                    setCountData(finalMaxScoreIdx, Integer.parseInt(label));
                    try {
                        String tobePosted = "{\"prediction\":" + Arrays.toString(outTensor.getDataAsFloatArray()) + ",\"label\":" + label + ",\"UUID\":\"" + id(getApplicationContext()) + "\"}";
                        post(tobePosted);
                        Toast.makeText(getApplicationContext(), "Your label submitted!",
                                Toast.LENGTH_SHORT).show();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                else {
                    input.setText("");
                    Toast.makeText(this, "Please provide a valid label!", Toast.LENGTH_SHORT).show();
                }
            })
            .setNegativeButton("Cancel", (dialog, id) -> {
                dialog.cancel();
                Toast.makeText(getApplicationContext(),"Cancelled",
                        Toast.LENGTH_SHORT).show();
            });
        AlertDialog alert = builder.create();
        alert.setTitle("Label input");
        alert.setCanceledOnTouchOutside(false);
        alert.show();
    }


    private static class UpdateModule extends AsyncTask<String, Integer, Integer>{
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressDialog.show();
        }
        @Override
        protected Integer doInBackground(String... url) {
            try {
                URL url_ = new URL("https://fl-mnist-server.herokuapp.com/update");
                HttpURLConnection urlConnection = (HttpURLConnection) url_.openConnection();
                urlConnection.setRequestMethod("GET");
                urlConnection.connect();

                File file = new File(dir,"updated_model.pt");
                FileOutputStream fileOutput = new FileOutputStream(file);
                InputStream inputStream = urlConnection.getInputStream();
                int totalSize = urlConnection.getContentLength();
                int downloadedSize = 0;

                byte[] buffer = new byte[1024];
                int bufferLength;
                while((bufferLength = inputStream.read(buffer))>0){
                    fileOutput.write(buffer, 0, bufferLength);
                    downloadedSize += bufferLength;
                    publishProgress((int) (totalSize * 100 / downloadedSize));
                }
                fileOutput.close();
                module_layer_2 = Module.load(dir+"/updated_model.pt");
            } catch (IOException e) {
                e.printStackTrace();
                return -1;
            }
            return null;
        }
        protected void onProgressUpdate(Integer... progress){
            progressDialog.setProgress(progress[0]);
        }
//
        @Override
        protected void onPostExecute(Integer result) {
            progressDialog.dismiss();
            if(result!=null){
                Toast.makeText(context, "Download Error!", Toast.LENGTH_SHORT).show();
            }
            else{
                Toast.makeText(context, "Downloaded Successfully!", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void getAccuracy(){
        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            int totalCount = 0;
            for(int count : Objects.requireNonNull(labelData.get(i))){
                totalCount += count;
            }
            dataPoints[i] = new DataPoint(i, (totalCount==0)?0:(Objects.requireNonNull(labelData.get(i))[i]/(totalCount*1.0))*100);
        }
        System.out.println(Arrays.toString(dataPoints));

        AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);
        final View accuracyLayout = getLayoutInflater().inflate(R.layout.accuracy_layout, null);
        GraphView graphview = accuracyLayout.findViewById(R.id.accuracy_plot);
        TextView textView = accuracyLayout.findViewById(R.id.accuracy_text);
        textView.setText(R.string.accuracy);
        LineGraphSeries<DataPoint> series = new LineGraphSeries<>(dataPoints);
        graphview.addSeries(series);
        Viewport graphViewport = graphview.getViewport();
        // set scrolling and zooming
        graphViewport.setScalable(true);
        graphViewport.setScrollable(true);
        graphViewport.setScalableY(true);
        graphViewport.setScrollableY(true);
        // set manual X bounds
        graphViewport.setXAxisBoundsManual(true);
        graphViewport.setMinX(0);
        graphViewport.setMaxX(9);
        // set manual Y bounds
        graphViewport.setYAxisBoundsManual(true);
        graphViewport.setMinY(0);
        graphViewport.setMaxY(100);

        graphview.setBackgroundColor(0x3700B3);

        series.setOnDataPointTapListener((series1, dataPoint) -> Toast.makeText(getApplicationContext(), dataPoint.getX()+": "+dataPoint.getY()+"%", Toast.LENGTH_SHORT).show());

        alertDialog.setView(accuracyLayout);
        AlertDialog alert = alertDialog.create();
        alert.show();
    }

    private void getF1Score(){
        int[][] data = new int[MNIST_LEN][MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            data[i] = labelData.get(i).clone();
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

        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];
        for(int i=0; i<MNIST_LEN; i++){
            double p = precision[i]==0?0:data[i][i]/(precision[i]*1.0);
            double r = recall[i]==0?0:data[i][i]/(recall[i]*1.0);
            double f1score = (p+r)==0?0:(2*p*r)/(p+r);
            dataPoints[i] = new DataPoint(i, f1score*100);
        }
        AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);
        final View f1Layout = getLayoutInflater().inflate(R.layout.accuracy_layout, null);
        GraphView graphview = f1Layout.findViewById(R.id.accuracy_plot);
        TextView textView = f1Layout.findViewById(R.id.accuracy_text);
        textView.setText(R.string.f1_score);
        LineGraphSeries<DataPoint> series = new LineGraphSeries<>(dataPoints);
        graphview.addSeries(series);
        Viewport graphViewport = graphview.getViewport();
        // set scrolling and zooming
        graphViewport.setScalable(true);
        graphViewport.setScrollable(true);
        graphViewport.setScalableY(true);
        graphViewport.setScrollableY(true);
        // set manual X bounds
        graphViewport.setXAxisBoundsManual(true);
        graphViewport.setMinX(0);
        graphViewport.setMaxX(9);
        // set manual Y bounds
        graphViewport.setYAxisBoundsManual(true);
        graphViewport.setMinY(0);
        graphViewport.setMaxY(100);

        graphview.setBackgroundColor(0x3700B3);

        series.setOnDataPointTapListener((series1, dataPoint) -> Toast.makeText(getApplicationContext(), dataPoint.getX()+": "+dataPoint.getY()+"%", Toast.LENGTH_SHORT).show());

        alertDialog.setView(f1Layout);
        AlertDialog alert = alertDialog.create();
        alert.show();
    }

    public synchronized static String id(Context context){
        if (uniqueID == null) {
            SharedPreferences sharedPrefs = context.getSharedPreferences(
                    PREF_UNIQUE_ID, Context.MODE_PRIVATE);
            uniqueID = sharedPrefs.getString(PREF_UNIQUE_ID, null);
            if (uniqueID == null) {
                uniqueID = UUID.randomUUID().toString();
                SharedPreferences.Editor editor = sharedPrefs.edit();
                editor.putString(PREF_UNIQUE_ID, uniqueID);
                editor.apply();
            }
        }
        return uniqueID;
    }

    public synchronized static void setTotalCount(Context context){
//        String todayCount = date+" "+TOTAL_COUNT;
        SharedPreferences sharedPrefs1 = context.getSharedPreferences(
                TOTAL_COUNT, Context.MODE_PRIVATE);
        String total_count = sharedPrefs1.getString(TOTAL_COUNT, null);
        int count = total_count==null?1:Integer.parseInt(total_count)+1;
        System.out.println("count "+count);
        SharedPreferences.Editor editor = sharedPrefs1.edit();
        editor.putString(TOTAL_COUNT, count+"");
        editor.apply();
    }

    public synchronized static void setTotalCorrect(Context context){
//        String todayCorrect = date+" "+TOTAL_CORRECT;
        SharedPreferences sharedPrefs2 = context.getSharedPreferences(
                TOTAL_CORRECT, Context.MODE_PRIVATE);
        String total_correct = sharedPrefs2.getString(TOTAL_CORRECT, null);
        int correct = total_correct==null?1:Integer.parseInt(total_correct)+1;
        System.out.println("correct "+correct);
        SharedPreferences.Editor editor2 = sharedPrefs2.edit();
        editor2.putString(TOTAL_CORRECT, correct+"");
        editor2.apply();
    }

    public static void setCountData(int prediction, int label){
        if(labelData.containsKey(prediction)) {
            labelData.get(label)[prediction]++;
        }
        for(int i=0; i<10; i++){
            System.out.println("count "+i+" "+Arrays.toString(labelData.get(i)));
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(dir+"/labelCount.txt");
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
            objectOutputStream.writeObject(labelData);
            objectOutputStream.close();
            fileOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        try {
            FileInputStream fileInputStream = new FileInputStream(dir+"/labelCount.txt");
            ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
            labelData = (HashMap)objectInputStream.readObject();
            objectInputStream.close();
            fileInputStream.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}