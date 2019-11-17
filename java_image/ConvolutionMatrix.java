import java.awt.image.BufferedImage;

public class ConvolutionMatrix{
  public static final int SIZE = 3;

  public double[][] Matrix;

  public ConvolutionMatrix(int size){
    Matrix = new double[size][size];
  }

  public void applyConfig(double[][] config){
    for (int x = 0; x < SIZE; x++){
      for (int y = 0; y < SIZE; y++){
        Matrix[x][y] = config[x][y];
      }
    }
  }

  public static BufferedImage filt3x3(BufferedImage src, ConvolutionMatrix matrix){
    int width = src.getWidth();
    int height = src.getHeight();
    BufferedImage result = new BufferedImage(width, height, src.getType());

    int A, R, G, B;
    int sumR, sumG, sumB;
    int[][] pixels = new int[SIZE][SIZE];

    for (int x = 0; x < width - 2; x++){
      for (int y = 0; y < height - 2; y++){
        // get pixel Matrix
        for (int i = 0; i < SIZE; i++){
          for (int j = 0; j < SIZE; j++){
            pixels[i][j] = src.getRGB(x+i, y+j);
          }
        }

        // get alpha
        A = (pixels[1][1] >> 24) & 0xFF;

        // init color sum
        sumR = 0;
        sumG = 0;
        sumB = 0;

        // 2D convolution
        for (int i = 0; i < SIZE; i++){
          for (int j = 0; j < SIZE; j++){
            sumR += (((pixels[i][j] >> 16) & 0xFF) * matrix.Matrix[i][j]);
            sumG += (((pixels[i][j] >>  8) & 0xFF) * matrix.Matrix[i][j]);
            sumB += (((pixels[i][j] >>  0) & 0xFF) * matrix.Matrix[i][j]);
          }
        }

        // get final R/G/B
        R = (int)sumR;
        if (R < 0){
          R = 0;
        }
        if (R > 255){
          R = 255;
        }

        G = (int)sumG;
        if (G < 0){
          G = 0;
        }
        if (G > 255){
          G = 255;
        }

        B = (int)sumB;
        if (B < 0){
          B = 0;
        }
        if (B > 255){
          B = 255;
        }

        // apply result
        result.setRGB(x+1,y+1,A*16777216+G*65536+R*256+B);

      }
    }

    return result;
  }
}
