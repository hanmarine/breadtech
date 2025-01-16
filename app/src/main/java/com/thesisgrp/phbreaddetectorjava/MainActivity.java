package com.thesisgrp.phbreaddetectorjava;

import static java.text.Normalizer.normalize;

import org.pytorch.*;
import org.pytorch.torchvision.TensorImageUtils;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.LifecycleOwner;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    TextView textView;
    private int REQUEST_CODE_PERMISSION = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};
    List<String> labels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);
        labels = loadClasses("labels.txt");

        if(!checkPermissions()){
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION);
        }

        LoadTorchModule("final.ptl");
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() ->{
            try{
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCamera(cameraProvider);
            } catch(ExecutionException | InterruptedException e){
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean checkPermissions() {
        for(String permission : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    // Starting the camera
    Executor executor = Executors.newSingleThreadExecutor();
    void startCamera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(224,224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                analyzeImage(image, rotation);
                image.close();
            }
        });

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);
    }


    // Load the module
    Module module;
    void LoadTorchModule(String fileName){
        File modelFile = new File(this.getFilesDir(), fileName);
        try{
            if(!modelFile.exists()){
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int bytesRead = -1;
                while((bytesRead = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void analyzeImage(ImageProxy image, int rotation){
        @SuppressLint("UnsafeOptInUsageError") Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(image.getImage(), rotation, 640,640,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        System.out.println("Tensor input shape " + Arrays.toString(inputTensor.shape()));

        // converting tensor to tuple
        IValue output = module.forward(IValue.from(inputTensor));
        IValue[] outputTuple = output.toTuple();
        IValue firstElement = outputTuple[1];
        Tensor outputTensor = firstElement.toTensor();
        System.out.println("Tensor output shape " + Arrays.toString(outputTensor.shape()));

        // Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // get the scores:
        float[] scores = outputTensor.getDataAsFloatArray();
        int numClasses = 16; // no of breads
        int numBoxes = 32;
        int gridHeight = 160;
        int gridWidth = 160;
        int numElementsPerCell = 1 + numClasses + 4;

        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;

        for(int i = 0; i <= numBoxes * gridHeight * gridWidth; i++){
            int offset = i * numElementsPerCell;

            for(int j = 0; j <= numClasses; j++){
                float score = scores[offset+1+j];
                if(score > maxScore){
                    maxScore = score;
                    maxScoreIdx = j;
                }
            }
        }
//        float[] scores = outputTensor.getDataAsFloatArray();
//        System.out.println("Scores " + scores.length);
//        float maxScore = -Float.MAX_VALUE;
//        int maxScoreIdx = -1;
//
//        for(int i=0; i<Math.min(scores.length, 17); i++){
//            if(scores[i]>maxScore){
//                maxScore = scores[i];
//                maxScoreIdx = i;
//            }
//        }

        String classResult = labels.get(maxScoreIdx);
        Log.v("Torch", "Detected - " + classResult);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textView.setText(classResult);
            }
        });
    }

    // 17 classes
    List<String> loadClasses(String fileName){
        List<String> classes = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(fileName)));
            String line;
            while ((line = br.readLine()) != null){
                classes.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classes;
    }
}