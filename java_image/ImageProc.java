
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

		// Modify image: swap green and red channels
		int width = image.getWidth();
		int height = image.getHeight();
		int pixel;
		int A, R, G, B;
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
			{
				pixel = image.getRGB(i,j);
				A = (pixel >> 24) & 0xFF;
				R = (pixel >> 16) & 0xFF;
				G = (pixel >>  8) & 0xFF;
				B = (pixel >>  0) & 0xFF;
				image.setRGB(i,j,A*16777216+G*65536+R*256+B); // reverse Green & Red channels
			}

		// Display the modified image
    ImageIcon icon=new ImageIcon(image);
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
