package com.example.nan.imageproc;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.io.FileNotFoundException;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.Color;

public class MainActivity extends AppCompatActivity {

    public static final int SIZE = 3;
    private static final String TAG = "ImageProc";

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    TextView textTargetUri;
    ImageView targetImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button buttonLoadImage = (Button)findViewById(R.id.loadimage);
        textTargetUri = (TextView)findViewById(R.id.targeturi);
        targetImage = (ImageView)findViewById(R.id.targetimage);

        buttonLoadImage.setOnClickListener(new Button.OnClickListener(){

            @Override
            public void onClick(View arg0) {
                // TODO Auto-generated method stub
                Intent intent = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 0);
            }});
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // TODO Auto-generated method stub
        super.onActivityResult(requestCode, resultCode, data);

        double[][] matrix = {{1,2,1},{0,0,0},{-1,-2,-1}};

        if (resultCode == RESULT_OK){
            Uri targetUri = data.getData();
            textTargetUri.setText(targetUri.toString());
            Bitmap bitmap;
            try {
                bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(targetUri));
                bitmap = computeConvolution3x3_fast(bitmap, matrix);
                targetImage.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public static Bitmap computeConvolution3x3_fast(Bitmap src, double[][] matrix) {
        int width = src.getWidth();
        int height = src.getHeight();
        Bitmap result = Bitmap.createBitmap(width, height, src.getConfig());
        int[] img_values = new int[width * height];
        float[] color_values = new float[width * height * 3];
        int[] result_values = new int[width * height];

        int A, R, G, B;
        double R_temp, G_temp, B_temp;
        int[][] pixels = new int[SIZE][SIZE];
        A = 255;

        src.getPixels(img_values, 0, src.getWidth(), 0, 0, src.getWidth(), src.getHeight());
        for (int i = 0; i < img_values.length; i++) {
            final int val = img_values[i];
            color_values[i * 3 + 0] = ((val >> 16) & 0xFF); //red
            color_values[i * 3 + 1] = ((val >> 8) & 0xFF); //green
            color_values[i * 3 + 2] = (val & 0xFF); //blue
        }

        for(int y = 0; y < height - 2; ++y) {
            for(int x = 0; x < width - 2; ++x) {

                R_temp = G_temp = B_temp = 0;

                // get sum of RGB on matrix
                for(int i = 0; i < SIZE; ++i) {
                    for(int j = 0; j < SIZE; ++j) {
                        R_temp += color_values[3*((y + i) * width + x + i)] * matrix[i][j];
                        G_temp += color_values[3*((y + i) * width + x + i) + 1] * matrix[i][j];
                        B_temp += color_values[3*((y + i) * width + x + i) + 2] * matrix[i][j];
                    }
                }

                // get final Red
                //R = (int)(sumR / matrix.Factor + matrix.Offset);
                R = (int)R_temp ;
                if(R < 0) { R = 0; }
                else if(R > 255) { R = 255; }

                // get final Green
                G = (int)(G_temp);
                if(G < 0) { G = 0; }
                else if(G > 255) { G = 255; }

                // get final Blue
                B = (int)(B_temp);
                if(B < 0) { B = 0; }
                else if(B > 255) { B = 255; }

                // apply new pixel
                //result.setPixel(x + 1, y + 1, Color.argb(A, R, G, B));
                result_values[(y + 1) * width + x + 1] = Color.argb(A, R, G, B);
            }
        }
        result.setPixels(result_values, 0, src.getWidth(), 0, 0, src.getWidth(), src.getHeight());

        // final image
        return result;
    }

}
