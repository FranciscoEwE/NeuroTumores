package com.example.neurotumores;


import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.neurotumores.ml.ClassifierBrainTumor;
import com.example.neurotumores.ml.ClassifierBrainTumorModel2;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;




public class MainActivity extends AppCompatActivity {

    Button Camara, CargarImagen;
    ImageView visualizador;
    TextView prediccion;
    int imageSize = 250;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Camara = findViewById(R.id.Camara);
        CargarImagen = findViewById(R.id.CargarImagen);

        prediccion = findViewById(R.id.prediccion);
        visualizador = findViewById(R.id.imageView);

        Camara.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        CargarImagen.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap imagen){
        try {
            //ClassifierBrainTumorModel2 modelo = ClassifierBrainTumorModel2.newInstance(getApplicationContext());
            ClassifierBrainTumor modelo = ClassifierBrainTumor.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 250, 250,3}, DataType.FLOAT32);
            // Creates inputs for reference.
            //TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1,64, 64, 1}, DataType.FLOAT32);

            Log.d("shape 3", inputFeature0.getBuffer().toString());
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());


            int[] intValues = new int[imageSize * imageSize];
            imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);






            // Runs model inference and gets result.
            //ClassifierBrainTumorModel2.Outputs outputs = modelo.process(inputFeature0);
            ClassifierBrainTumor.Outputs outputs = modelo.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Pituitary","Meningioma","No tumor","Glioma"};
            prediccion.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            modelo.close();
        } catch (IOException e) {
            prediccion.setText("No se pudo clasificar");

        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        Bitmap imagen;
        if(resultCode == RESULT_OK){
            if(requestCode == 3){

                Bitmap image = (Bitmap) data.getExtras().get("data");
                visualizador.setImageBitmap(image);
                imagen = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                Log.d("shape 1", TensorImage.fromBitmap(imagen).getBuffer().toString());
                classifyImage(imagen);

            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                visualizador.setImageBitmap(image);
                imagen = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                Log.d("shape 1", TensorImage.fromBitmap(imagen).getBuffer().toString());
                classifyImage(imagen);

            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }
}