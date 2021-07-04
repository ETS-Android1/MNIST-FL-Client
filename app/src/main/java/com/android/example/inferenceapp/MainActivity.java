package com.android.example.inferenceapp;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.icu.text.SimpleDateFormat;
import android.os.AsyncTask;
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
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;

import static android.Manifest.permission.INTERNET;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class MainActivity extends AppCompatActivity {
    private CanvasView canvasView;
    private EditText input;
    private Module module_layer_1;
    private static Module module_layer_2;
    private AlertDialog.Builder builder;
    private static ProgressDialog progressDialog;

    private static String uniqueID = null;
    private static final String PREF_UNIQUE_ID = "PREF_UNIQUE_ID";
    private static final String TOTAL_COUNT = "TOTAL_COUNT";
    private static final String TOTAL_CORRECT = "TOTAL_CORRECT";

    private static final int PERMISSION_REQUEST_CODE = 200;

    private String label = "";
    private static String date;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        canvasView = findViewById(R.id.main_canvas);
        Button predict = findViewById(R.id.predict);
        Button update = findViewById(R.id.update_model);
        date = new SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(new Date());
        builder = new AlertDialog.Builder(this);
        input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);

        if(!checkPermission()){
            requestPermission();
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

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch(item.getItemId()){
            case R.id.update:
                update();
                return true;
            case R.id.graph:
                Toast.makeText(this, "Graph Selected!", Toast.LENGTH_SHORT).show();
                return true;
            case R.id.feedback:
                Toast.makeText(this, "Feedback Selected!", Toast.LENGTH_SHORT).show();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private boolean checkPermission() {
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int result2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        int result3 = ContextCompat.checkSelfPermission(getApplicationContext(), INTERNET);
        return result1== PackageManager.PERMISSION_GRANTED && result2== PackageManager.PERMISSION_GRANTED
                && result3== PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission(){
        ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET}, PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0) {

                    boolean writeAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean readAccepted = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                    boolean internetAccepted = grantResults[2] == PackageManager.PERMISSION_GRANTED;

                    if (writeAccepted && readAccepted && internetAccepted)
                        Toast.makeText(this, "Permission Granted, Now app can access storage", Toast.LENGTH_SHORT).show();
                    else if(!writeAccepted || !readAccepted || !internetAccepted) {
                        Toast.makeText(this, "Permission Denied, App cannot access storage", Toast.LENGTH_SHORT).show();

                        if (shouldShowRequestPermissionRationale(WRITE_EXTERNAL_STORAGE)) {
                            showMessageOKCancel("You need to allow access to all the permissions",
                                    (dialog, which) -> requestPermissions(new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, INTERNET},
                                            PERMISSION_REQUEST_CODE));
                            return;
                        }
                    }
                }
                break;
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
        canvasView.screenShot();
        String storageDirectory = Environment.getExternalStorageDirectory().getPath();
        Bitmap bitmap = BitmapFactory.decodeFile(storageDirectory+"/img.jpg");
        FloatBuffer inputBuff = Tensor.allocateFloatBuffer(bitmap.getHeight() * bitmap.getWidth());

        final double GS_RED = 0.299;
        final double GS_GREEN = 0.587;
        final double GS_BLUE = 0.114;

        for(int y=0; y<bitmap.getHeight(); y++){
            for(int x=0; x<bitmap.getWidth(); x++){
                int pixel = bitmap.getPixel(x, y);
//                int A = Color.alpha(pixel);
                int R = Color.red(pixel);
                int G = Color.green(pixel);
                int B = Color.blue(pixel);
                double val = 255-(R * GS_RED + G * GS_GREEN + B * GS_BLUE);
//                Log.d("val ", String.valueOf(val));
                val /= 255;
                inputBuff.put((float) val);
            }
        }


        Tensor inputTensor = Tensor.fromBlob(inputBuff, new long[]{1, 1, bitmap.getHeight(), bitmap.getWidth()});

//        For Debugging input
        float[] toPrint = inputTensor.getDataAsFloatArray();
        for(int i=0; i<bitmap.getHeight()*bitmap.getWidth(); i+=bitmap.getWidth()){
            Log.d("input ", Arrays.toString(Arrays.copyOfRange(toPrint, i, i+28)));
        }

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
                if(!label.trim().equals("") && label.length()==1) {
                    input.setText("");
                    setTotalCount(getApplicationContext());
                    if (Integer.parseInt(label) == finalMaxScoreIdx) {
                        setTotalCorrect(getApplicationContext());
                    }
                    try {
                        String tobePosted = "{\"prediction\":" + Arrays.toString(outTensor.getDataAsFloatArray()) + ",\"label\":" + label + ",\"UUID\":\"" + id(getApplicationContext()) + "\"}";
                        System.out.println(tobePosted.substring(tobePosted.length() - 50));
                        post(tobePosted);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    Toast.makeText(getApplicationContext(), "Your label submitted!",
                            Toast.LENGTH_SHORT).show();
                }
                else {
                    input.setText("");
                }
            })
            .setNegativeButton("Cancel", (dialog, id) -> {
                //  Action for 'NO' Button
                dialog.cancel();
                Toast.makeText(getApplicationContext(),"Cancelled",
                        Toast.LENGTH_SHORT).show();
            });
        AlertDialog alert = builder.create();
        alert.setTitle("Label input");
        alert.show();
    }


    private static class UpdateModule extends AsyncTask<String, Integer, Integer>{
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressDialog.show();
//            showDialog(0);
        }
        @Override
        protected Integer doInBackground(String... url) {
            try {
                URL url_ = new URL("https://fl-mnist-server.herokuapp.com/update");
                HttpURLConnection urlConnection = (HttpURLConnection) url_.openConnection();
                urlConnection.setRequestMethod("GET");
                urlConnection.connect();

                File storageDirectory = Environment.getExternalStorageDirectory();
                File file = new File(storageDirectory,"updated_model.pt");
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
                module_layer_2 = Module.load(Environment.getExternalStorageDirectory().getPath()+"/updated_model.pt");
            } catch (IOException e) {
                e.printStackTrace();
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
        }
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
        String todayCount = date+" "+TOTAL_COUNT;
        SharedPreferences sharedPrefs1 = context.getSharedPreferences(
                todayCount, Context.MODE_PRIVATE);
        String total_count = sharedPrefs1.getString(todayCount, null);
        int count = total_count==null?1:Integer.parseInt(total_count)+1;
        System.out.println("count "+count);
        SharedPreferences.Editor editor = sharedPrefs1.edit();
        editor.putString(todayCount, count+"");
        editor.apply();
    }

    public synchronized static void setTotalCorrect(Context context){
        String todayCorrect = date+" "+TOTAL_CORRECT;
        SharedPreferences sharedPrefs2 = context.getSharedPreferences(
                todayCorrect, Context.MODE_PRIVATE);
        String total_correct = sharedPrefs2.getString(todayCorrect, null);
        int correct = total_correct==null?1:Integer.parseInt(total_correct)+1;
        System.out.println("correct "+correct);
        SharedPreferences.Editor editor2 = sharedPrefs2.edit();
        editor2.putString(todayCorrect, correct+"");
        editor2.apply();
    }
}