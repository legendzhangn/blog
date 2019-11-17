import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class ImageProc {

	public static void main(String[] args) {
		File file = new File("Lenna.png");
    BufferedImage image = null;
    BufferedImage image2 = null;

		// Read image
    try
    {
        image = ImageIO.read(file);
    }
    catch (IOException e)
    {
        e.printStackTrace();
    }
    System.out.println("done");


    // 2D convolution for the image
    double[][] config = {{1,2,1}, {0,0,0}, {-1,-2,-1}};
    ConvolutionMatrix imageConv = new ConvolutionMatrix(3);
    imageConv.applyConfig(config);
    image2 = imageConv.filt3x3(image, imageConv);

		// Display the modified image
    ImageIcon icon=new ImageIcon(image2);
    JFrame frame=new JFrame();
    frame.setLayout(new FlowLayout());
    frame.setSize(image.getWidth(),image.getHeight()); //Window.setSize(int width, int height)
    JLabel lbl=new JLabel();
    lbl.setIcon(icon);
    frame.add(lbl);
    frame.setVisible(true);
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}

}
