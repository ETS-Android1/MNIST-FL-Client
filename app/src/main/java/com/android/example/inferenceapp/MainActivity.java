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
import android.icu.text.DateFormat;
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
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
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
import com.jjoe64.graphview.series.LineGraphSeries;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLConnection;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import static android.Manifest.permission.INTERNET;
import static android.Manifest.permission.MANAGE_EXTERNAL_STORAGE;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    private CanvasView canvasView;
    private EditText input;
    private Module module_layer_1;
    private static Module module_layer_2;
    private AlertDialog.Builder builder;
    private static ProgressDialog progressDialog;
    private static Context context;
    private UpdateModule updateModule;

    private static String uniqueID = null;
    private static final String PREF_UNIQUE_ID = "PREF_UNIQUE_ID";
    private static final String CURRENT_MODEL = "CURRENT_MODEL";

    private static final int PERMISSION_REQUEST_CODE = 200;
    private static final int MNIST_LEN = 10;

    private String label = "";
    private static HashMap<Integer, int[]> labelData;

    private static final String PATH = Environment.getExternalStorageDirectory().getPath();
    private static final String dir = PATH+"/MNIST_FL";

    private String currModel;
    private String version;
    private Model currentModel;
//    private Set<String> modelList;
    private Set<Model> modelList;
//    private List<Model> modelList;
//    private boolean isModelSet = false;
//    private SimpleDateFormat simpleDateFormat;

    // TODO: initially no model will be there so prediction will be none
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        canvasView = findViewById(R.id.main_canvas);
        Button predict = findViewById(R.id.predict);
        Button update = findViewById(R.id.update_model);
        context = getApplicationContext();
//        String date = new SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(new Date());
        builder = new AlertDialog.Builder(this);
        input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
//        updateModule = new UpdateModule();

//        currModel = "";
        currentModel = null;
        labelData = new HashMap<>();
        for(int i=0; i<10; i++){
            labelData.put(i, new int[MNIST_LEN]);
        }

        if(!checkPermission()){
            requestPermission();
        }
//        File directory = new File(dir);
//        if(!directory.exists()){
//            directory.mkdir();
//        }

        progressDialog = new ProgressDialog(MainActivity.this);
        progressDialog.setMessage("Model updating...");
        progressDialog.setIndeterminate(false);
        progressDialog.setMax(100);
        progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        try {
            module_layer_1 = Module.load(assetFilePath(getApplicationContext(), "model_mnist_cnn_1.pt"));
//            module_layer_2 = Module.load(assetFilePath(getApplicationContext(), "model_mnist_cnn_2.pt"));
        }catch(IOException e){
            e.printStackTrace();
        }


        update.setOnClickListener(view -> {
            try {
                update();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        predict.setOnClickListener(view -> {
            try {
                recognize();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        initialSetup();

//        File models = new File(dir+"/Models");
//        File images = new File(dir+"/images");
//        boolean mkdirs1 = false, mkdirs2 = false, mkdirs3 = false;
//        if(!models.exists()){
//            mkdirs1 = models.mkdirs();
//        }
//        if(!images.exists()){
//            mkdirs2 = images.mkdirs();
//        }
//        File globalModel = new File(dir+"/Models/globalModel");
//        if(!globalModel.exists()){
//            mkdirs3 = globalModel.mkdirs();
//        }
//
//        Log.d("dir creation: ", mkdirs1+" "+mkdirs2+" "+mkdirs3);
//        SharedPreferences prefs = context.getSharedPreferences(
//                CURRENT_MODEL, Context.MODE_PRIVATE
//        );
//        currModel = prefs.getString(CURRENT_MODEL, null);
//        modelList = new ArrayList<>();
//        if(currModel==null){
//            currentModel = new Model("globalModel");
//            if(globalModel.exists() && currModel==null) {
//                updateModule = new UpdateModule();
//                String url = "https://firebasestorage.googleapis.com/v0/b/test-4acc6.appspot.com/o/Models%2Fmodel_mnist_cnn_2.pt?alt=media&token=fc00cd3f-eab4-486b-b76d-f14a7bd9d258";
//                updateModule.execute(url, dir + "/Models/globalModel");
//                currModel = dir + "/Models/globalModel";
//                modelList.add(currModel);
////            isModelSet = true;
//            }
//        }
//        modelList = new ArrayList<>();
//        setModelList();
//        readModelList();

//        Log.d("GModel", currModel==null?"null":currModel);
//        if(globalModel.exists() && currModel==null) {
//            updateModule = new UpdateModule();
//            String url = "https://firebasestorage.googleapis.com/v0/b/test-4acc6.appspot.com/o/Models%2Fmodel_mnist_cnn_2.pt?alt=media&token=fc00cd3f-eab4-486b-b76d-f14a7bd9d258";
//            updateModule.execute(url, dir + "/Models/globalModel");
//            currModel = dir + "/Models/globalModel";
//            modelList.add(currModel);
////            isModelSet = true;
//        }
//        else if(new File(currModel).exists()){
//            module_layer_2 = Module.load(currModel+"/updated_model.pt");
//        }
    }

    private void initialSetup(){
        File models = new File(dir+"/Models");
        File images = new File(dir+"/images");
        boolean mkdirs1 = false, mkdirs2 = false, mkdirs3 = false;
        if(!models.exists()){
            mkdirs1 = models.mkdirs();
        }
        if(!images.exists()){
            mkdirs2 = images.mkdirs();
        }
        File globalModel = new File(dir+"/Models/globalModel");
        if(!globalModel.exists()){
            mkdirs3 = globalModel.mkdirs();
        }

        Log.d("dir creation: ", mkdirs1+" "+mkdirs2+" "+mkdirs3);
        SharedPreferences prefs = context.getSharedPreferences(
                CURRENT_MODEL, Context.MODE_PRIVATE
        );
        modelList = new HashSet<>();
        currModel = prefs.getString(CURRENT_MODEL, null);
        File module = new File(globalModel.getPath()+"/updated_model.pt");
        if(currModel==null || currentModel==null || !module.exists()){
            currentModel = new Model("globalModel");
            if(!globalModel.exists()) mkdirs3 = globalModel.mkdirs();
            if(!module.exists()) {
                updateModule = new UpdateModule();

                // TODO: Global model is to be set in the server to the firebase

                String url = "https://firebasestorage.googleapis.com/v0/b/test-4acc6.appspot.com/o/Models%2Fmodel_mnist_cnn_2.pt?alt=media&token=fc00cd3f-eab4-486b-b76d-f14a7bd9d258";
                String modelPath = dir + "/Models/globalModel";
                updateModule.execute(url, modelPath);
                if(!updateModule.isCancelled()){
                    try {
                        module_layer_2 = Module.load(modelPath + "/updated_model.pt");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                currModel = modelPath;
                modelList.add(currentModel);
            }
        }
        else{
            currentModel = new Model(currModel);
            module_layer_2 = Module.load(currentModel.getmPath()+"/updated_model.pt");
        }
        setModelList();
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
            case R.id.choose_model:
                setModel();
                return true;
            case R.id.update:
                try {
                    update();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
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
//                    System.out.println(writeAccepted + " " + readAccepted + " " + internetAccepted + " " + isR + " " + manageAccepted);
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
//                HttpURLConnection connection = (HttpURLConnection) new URL("https://fl-mnist-server.herokuapp.com/train").openConnection();
                HttpURLConnection connection = (HttpURLConnection) new URL("http://192.168.43.218:5000/train").openConnection();
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

    private void update() throws InterruptedException {
        String datetime = getDateTime();
        getServerModelVersion("http://192.168.43.218:5000/version");
        String model_path = dir+"/Models/Model_"+version;
        File newModel = new File(model_path);
        if(!newModel.exists()){
            newModel.mkdir();
        }
        updateModule = new UpdateModule();
//        updateModule.execute("https://fl-mnist-server.herokuapp.com/update", model_path);
        updateModule.execute("http://192.168.43.218:5000/update", model_path);
        if(!updateModule.isCancelled()){
            try {
                module_layer_2 = Module.load(model_path + "/updated_model.pt");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
//        writeObject();
        currentModel.setCountData();
        labelData = new HashMap<>();
        for(int i=0; i<10; i++){
            labelData.put(i, new int[MNIST_LEN]);
        }
//        currModel = model_path;
//        modelList.add(currModel);

        modelList.add(new Model("Model_"+version));

//        modelList.add(new Model(currModel));
    }

    private void recognize() throws Exception {
        if(module_layer_2==null){
            Toast.makeText(context, "Please choose a model or download from the server!", Toast.LENGTH_SHORT).show();
            return;
        }
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
//                    setCountData(finalMaxScoreIdx, Integer.parseInt(label));
                    currentModel.incrementCount(finalMaxScoreIdx, Integer.parseInt(label));
                    try {
                        String tobePosted = "{\"prediction\":" + Arrays.toString(outTensor.getDataAsFloatArray()) + ",\"label\":" + label + ",\"UUID\":\"" + id(getApplicationContext()) + "\"}";
                        post(tobePosted);
                        storeImage(label, bitmap);
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

    private void storeImage(String label, Bitmap bmp){
        try (FileOutputStream out = new FileOutputStream(dir+"/images/"+label+"_"+getDateTime()+".jpg")) {
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setModel(){
        AlertDialog.Builder alertDialog = new AlertDialog.Builder(MainActivity.this);
        final View chooseLayout = getLayoutInflater().inflate(R.layout.model_chooser_layout, null);
        Spinner spinner = chooseLayout.findViewById(R.id.model_chooser);
//        Button chooseButton = chooseLayout.findViewById(R.id.model_button);
        TextView modelName = chooseLayout.findViewById(R.id.model_name);
//        modelName.setText(currModel);
        modelName.setText(currentModel.getmName());
        spinner.setOnItemSelectedListener(this);
//        spinner.setOnItemClickListener(this::onItemSelected);
        List<String> items = new ArrayList<>();
        for(Model model : modelList){
            String temp = model.getmName();
            items.add(temp);
        }
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, items);
        spinner.setAdapter(dataAdapter);
        alertDialog.setPositiveButton("Submit", (dialog, id) -> modelChooser(String.valueOf(spinner.getSelectedItem())))
        .setNegativeButton("Cancel", (dialog, id) -> {
            dialog.cancel();
            Toast.makeText(getApplicationContext(),"Cancelled",
                    Toast.LENGTH_SHORT).show();
        });
//        chooseButton.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                modelChooser(String.valueOf(spinner.getSelectedItem()));
//            }
//        });

        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        alertDialog.setView(chooseLayout);

        AlertDialog alert = alertDialog.create();
        alert.setTitle("Choose Model");
        alert.setCanceledOnTouchOutside(false);
        alert.show();
    }

    // spinner widget methods
    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        String item = adapterView.getItemAtPosition(i).toString();
//        try {
//            module_layer_2 = Module.load(dir + "/Models/" + item + "/updated_model.pt");
//            writeObject();
//            currModel = item;
//            setCurrentModel(context);
//            readObject();
//        } catch (Exception e) {
//            e.printStackTrace();
//            Toast.makeText(context, "Please make sure the model exists!", Toast.LENGTH_SHORT).show();
//        }
//        readObject();
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }

    private void modelChooser(String modelName){
        String oldModel = currentModel.getmName();
        try {
            module_layer_2 = Module.load(dir + "/Models/" + modelName + "/updated_model.pt");
//            writeObject();
            currentModel.setCountData();
//            currModel = dir + "/Models/" + modelName;
            currentModel = new Model(modelName);
            setCurrentModel(context);
//            readObject();
        } catch (Exception e) {
            e.printStackTrace();
            module_layer_2 = Module.load(dir + "/Models/" + oldModel + "/updated_model.pt");
//            writeObject();
            currentModel.setCountData();
//            currModel = dir + "/Models/" + modelName;
            currentModel = new Model(modelName);
            setCurrentModel(context);
            Toast.makeText(context, "Please make sure the model exists!", Toast.LENGTH_SHORT).show();
        }
//        readObject();
    }

    private void getServerModelVersion(String url) throws InterruptedException {
        StringBuilder response = new StringBuilder();
        Thread t = new Thread(() -> {
//            try {
//                URL url_ = new URL(url);
//                HttpURLConnection urlConnection = (HttpURLConnection) url_.openConnection();
//                urlConnection.setDoOutput(true);
////                urlConnection.setRequestMethod("GET");
//                urlConnection.connect();
//
//                BufferedReader in = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
//
////                StringBuilder response = new StringBuilder();
//                String inputLine;
//
////                String newLine = System.getProperty("line.separator");
//                while ((inputLine = in.readLine()) != null)
//                {
//                    response.append(inputLine);
//                }
//
//                in.close();
//                version = response.toString();
//                System.out.println("version "+version);
////                return response.toString();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
            HttpURLConnection conn = null;
            try {
                URL url_ = new URL(url);
                conn = (HttpURLConnection) url_.openConnection();
            } catch (IOException e) {
                e.printStackTrace();
            }

            try {
                assert conn != null;
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    version = reader.lines().collect(Collectors.joining("\n"));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("version "+version);
        });
        t.start();
        t.join();
//        return response.toString();
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
                URL url_ = new URL(url[0]);
//                URL url_ = new URL("https://fl-mnist-server.herokuapp.com/update");
                HttpURLConnection urlConnection = (HttpURLConnection) url_.openConnection();
                urlConnection.setRequestMethod("GET");
                urlConnection.connect();

                File file = new File(url[1],"/updated_model.pt");
                FileOutputStream fileOutput = new FileOutputStream(file);
                InputStream inputStream = urlConnection.getInputStream();
                int totalSize = urlConnection.getContentLength();
                int downloadedSize = 0;

                byte[] buffer = new byte[1024];
                int bufferLength;
                while((bufferLength = inputStream.read(buffer))>0){
                    fileOutput.write(buffer, 0, bufferLength);
                    downloadedSize += bufferLength;
                    publishProgress((totalSize * 100) / downloadedSize);
                }
                fileOutput.close();
//                module_layer_2 = Module.load(url[1]+"/updated_model.pt");
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
//        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];

        DataPoint[] dataPoints = currentModel.getAccuracy();

//        for(int i=0; i<MNIST_LEN; i++){
//            int totalCount = 0;
//            for(int count : Objects.requireNonNull(labelData.get(i))){
//                totalCount += count;
//            }
//            dataPoints[i] = new DataPoint(i, (totalCount==0)?0:(Objects.requireNonNull(labelData.get(i))[i]/(totalCount*1.0))*100);
//        }
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
//        int[][] data = new int[MNIST_LEN][MNIST_LEN];
//        for(int i=0; i<MNIST_LEN; i++){
//            data[i] = labelData.get(i).clone();
//        }
//        int[] precision = new int[MNIST_LEN];
//        int[] recall = new int[MNIST_LEN];
//        for(int i=0; i<MNIST_LEN; i++){
//            for(int j=0; j<MNIST_LEN; j++){
//                precision[i] += data[i][j];
//            }
//        }
//        for(int j=0; j<MNIST_LEN; j++){
//            for (int i = 0; i < MNIST_LEN; i++){
//                recall[j] += data[i][j];
//            }
//        }

//        DataPoint[] dataPoints = new DataPoint[MNIST_LEN];

        DataPoint[] dataPoints = currentModel.getF1Score();

//        for(int i=0; i<MNIST_LEN; i++){
//            double p = precision[i]==0?0:data[i][i]/(precision[i]*1.0);
//            double r = recall[i]==0?0:data[i][i]/(recall[i]*1.0);
//            double f1score = (p+r)==0?0:(2*p*r)/(p+r);
//            dataPoints[i] = new DataPoint(i, f1score*100);
//        }
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

//    public synchronized static void setTotalCount(Context context){
////        String todayCount = date+" "+TOTAL_COUNT;
//        SharedPreferences sharedPrefs1 = context.getSharedPreferences(
//                TOTAL_COUNT, Context.MODE_PRIVATE);
//        String total_count = sharedPrefs1.getString(TOTAL_COUNT, null);
//        int count = total_count==null?1:Integer.parseInt(total_count)+1;
//        System.out.println("count "+count);
//        SharedPreferences.Editor editor = sharedPrefs1.edit();
//        editor.putString(TOTAL_COUNT, count+"");
//        editor.apply();
//    }
//
//    public synchronized static void setTotalCorrect(Context context){
////        String todayCorrect = date+" "+TOTAL_CORRECT;
//        SharedPreferences sharedPrefs2 = context.getSharedPreferences(
//                TOTAL_CORRECT, Context.MODE_PRIVATE);
//        String total_correct = sharedPrefs2.getString(TOTAL_CORRECT, null);
//        int correct = total_correct==null?1:Integer.parseInt(total_correct)+1;
//        System.out.println("correct "+correct);
//        SharedPreferences.Editor editor2 = sharedPrefs2.edit();
//        editor2.putString(TOTAL_CORRECT, correct+"");
//        editor2.apply();
//    }

    public synchronized void setCurrentModel(Context context){
        SharedPreferences sharedPrefs = context.getSharedPreferences(
                CURRENT_MODEL, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPrefs.edit();
//        editor.putString(CURRENT_MODEL, currModel);
        editor.putString(CURRENT_MODEL, currentModel.getmName());
        editor.apply();
    }

    private void setModelList(){
        try{
            File modelDir = new File(dir+"/Models");
            File[] models = modelDir.listFiles();
            if(models==null||models.length==0){
                return;
            }
            for(File model : models){
//                if(!modelList.contains(model)) modelList.add(model.getPath());
                Model temp = new Model(model.getName());
                modelList.add(temp);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

//    private void writeModelList(){
//        try {
//            FileOutputStream fileOutputStream = new FileOutputStream(dir + "/models.txt");
//            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
//            if(modelList.size()==0){
////                modelList.add(dir + "/Models/globalModel");
//                modelList.add(new Model("globalModel"));
//            }
//            objectOutputStream.writeObject(modelList);
//            objectOutputStream.close();
//            fileOutputStream.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

//    private void readModelList(){
//        try {
//            File labelfile = new File(dir + "/models.txt");
//            if(labelfile.exists()) {
//                FileInputStream fileInputStream = new FileInputStream(dir + "/models.txt");
//                ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
//                modelList = (HashSet) objectInputStream.readObject();
//                objectInputStream.close();
//                fileInputStream.close();
//            }
//        } catch (IOException | ClassNotFoundException e) {
//            e.printStackTrace();
//            writeModelList();
//        }
//    }

    private void setCountData(int prediction, int label){
//        System.out.println("label data "+labelData.toString());
        if(labelData.isEmpty()){
            labelData = new HashMap<>();
            for(int i=0; i<10; i++){
                labelData.put(i, new int[MNIST_LEN]);
            }
        }
        if(labelData.containsKey(prediction)) {
            Objects.requireNonNull(labelData.get(label))[prediction]++;
        }
        for(int i=0; i<10; i++){
            System.out.println("count "+i+" "+Arrays.toString(labelData.get(i)));
        }
    }

    private void writeObject(){
        try {
//            FileOutputStream fileOutputStream = new FileOutputStream(dir + "/Models/" + currModel + "/labelCount.txt");
//            FileOutputStream fileOutputStream = new FileOutputStream(currModel + "/labelCount.txt");
            FileOutputStream fileOutputStream = new FileOutputStream(currentModel.getmPath() + "/labelCount.txt");
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
            objectOutputStream.writeObject(labelData);
            objectOutputStream.close();
            fileOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void readObject() {
        try {
//            File labelfile = new File(dir + "/Models/" + currModel + "/labelCount.txt");
//            File labelfile = new File(currModel + "/labelCount.txt");
            File labelfile = new File(currentModel.getmPath() + "/labelCount.txt");
            if(labelfile.exists()) {
//                FileInputStream fileInputStream = new FileInputStream(dir + "/Models/" + currModel + "/labelCount.txt");
//                FileInputStream fileInputStream = new FileInputStream(currModel + "/labelCount.txt");
                FileInputStream fileInputStream = new FileInputStream(currentModel.getmPath() + "/labelCount.txt");
                ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
                labelData = (HashMap) objectInputStream.readObject();
                for(int i=0; i<MNIST_LEN; i++){
                    System.out.println("read "+(i+1)+" "+Arrays.toString(labelData.get(i)));
                }
                objectInputStream.close();
                fileInputStream.close();
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
//        writeObject();
        currentModel.setCountData();
        System.out.println("model "+currentModel.getmName());
        setCurrentModel(context);
    }

    @Override
    protected void onResume() {
        super.onResume();
        SharedPreferences prefs = context.getSharedPreferences(
                CURRENT_MODEL, Context.MODE_PRIVATE
        );
        currModel = prefs.getString(CURRENT_MODEL, null);
        File globalModel = new File(dir+"/Models/globalModel");
        File module = new File(globalModel.getPath()+"/updated_model.pt");
        if(currModel==null || !globalModel.exists() || !module.exists()){
            initialSetup();
        }
        else{
            currentModel = new Model(currModel);
            module_layer_2 = Module.load(currentModel.getmPath()+"/updated_model.pt");
            readObject();
            setModelList();
        }
    }

    private String getDateTime() {
        @SuppressLint("SimpleDateFormat") DateFormat dateFormat = new SimpleDateFormat("dd-MM-yyyy_HH:mm:ss");
        Date date = new Date();
        return dateFormat.format(date);
    }
}