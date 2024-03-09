import os
import cv2
import numpy as np
import math
import argparse

class ShipDetection:

    def __init__(   self, 
                    opt_args,
                    image_path="three_ships_horizon.JPG",
                    output_image_path="./results/three_ships_boxed.tiff"
                    ):
        # set self variables
        self.image_path = image_path
        self.output_image_path = output_image_path
        self.opt_args = opt_args

    def create_folder(self, path="results"):
        """Create new folder

        Args:
            path (str, optional): path of created folder. Defaults to "results".
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def line_detection(self, edges, threshold=200):
        """Detect r, theta values that satisfy threshold value with HoughLines transform

        Args:
            edges (img_arr): Canny edge image
            threshold (int, optional): Threshold to filter lines for HoughLines function. Defaults to 200.

        Returns:
            _type_: lines (r, theta) values
        """
        # This returns an array of r and theta values
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)

        return lines

    
    def rotate_image(self, image, rotated_angle_degree):
        """Rotate source image with a specific degree angle

        Args:
            image (np_array): source image for rotation
            rotated_angle_degree (float): rotation angle in degree

        Returns:
            np_array, np_array: rotated image array, Rotation matrix
        """
        # Rotate the image to level the horizon
        height, width = image.shape[:2]

        # get roration matrix
        M = cv2.getRotationMatrix2D((width // 2, height // 2), rotated_angle_degree, 1)
        # rotate src image with rotation matrix
        rotated_image = cv2.warpAffine(image, M, (width, height))

        return rotated_image, M

    def rotate_image_to_be_level(self, lines, image, thres_dist=500):
        """Rotate source image to be level

        Args:
            lines (_type_): lines after implementing HoughLines transform
            image (_type_): source image to be rotated
            thres_dist (int, optional): threshold distance for filtering robust lines. Defaults to 500.

        Returns:
            _type_: _description_
        """
        thetas = []
        # radius = []
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            if r > thres_dist:
                thetas.append(theta)
                # radius.append(r)

        # get mean rotation angle
        mean_theta_rad = sum(thetas)/len(thetas)
        mean_theta_degree = mean_theta_rad/math.pi*180

        # identify rotation angle
        rotated_angle_degree  = 90-mean_theta_degree
        rotated_angle_radian = math.radians(rotated_angle_degree)

        # rotate image
        rotated_image, _ = self.rotate_image(image, rotated_angle_degree)

        return rotated_image, rotated_angle_degree


    def get_largest_rotated_rect(self, w, h, angle):
        """ Get largest rotated rectangle inside rotated image without blank pixels.
            Given a rectangle of size wxh that has been rotated by 'angle' (in
            radians), computes the width and height of the largest possible
            axis-aligned rectangle within the rotated rectangle.

        Args:
            w (int): width of source image before rotating
            h (int): height of source image before rotating
            angle (float): rotated angle in radian.

        Returns:
            float, float: largest width, largest height
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )
    
    def crop_image_around_center(self, image, width, height):
        """Crop image around center

        Args:
            image (np array): soruce image for cropping
            width (int): width for cropping from center
            height (int): height for cropping from center

        Returns:
            np array: cropped image
        """
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


    def extract_contours(self, image_rotated_cropped,
                                bilateral_kernel_size=9,
                                bilateral_sigma_color=75,
                                bilateral_sigma_space=75,
                                morpho_kernel_size = (3,3),
                                opening_iter=3,
                                dilation_iter=3,
                                upper_hsv =[120, 116, 192],
                                lower_hsv =[106, 44, 142],
                                thres_contour_area = 750
                                ):
        """Extract object contours based on contour areas

        Args:
            image_rotated_cropped (np array): source image
            bilateral_kernel_size (int, optional): bilarteral kernel size. Defaults to 9.
            bilateral_sigma_color (int, optional): bilarteral sigma color. Defaults to 75.
            bilateral_sigma_space (int, optional): bilarteral sigma space. Defaults to 75.
            morpho_kernel_size (tuple, optional): morphology kernel size. Defaults to (3,3).
            opening_iter (int, optional): iteration for opening morphology. Defaults to 3.
            dilation_iter (int, optional): iteration for dilation morphology. Defaults to 3.
            upper_hsv (list, optional): upper threshold for hsv color space. Defaults to [120, 116, 192].
            lower_hsv (list, optional): lower threshold for hsv color space. Defaults to [106, 44, 142].
            thres_contour_area (int, optional): threshold for filtering based on contour area, 
                                                    area < thres will be removed. Defaults to 750.

        Returns:
            _type_: object contours
        """
        # use bilateral to remove noise, and enhance sharp edges
        blur = cv2.bilateralFilter( image_rotated_cropped,
                                    bilateral_kernel_size,
                                    bilateral_sigma_color,
                                    bilateral_sigma_space)

        # convert from BGR to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # note: these values are tunning with trackbars in HSV space first,
        lower_hsv = np.array(lower_hsv)
        upper_hsv = np.array(upper_hsv)

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        kernel = np.ones(morpho_kernel_size, np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=opening_iter)

        dilation = cv2.dilate(opening,kernel,iterations = dilation_iter)

        # Find contours from the edged image
        contours, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out the contours of the ships
        object_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > thres_contour_area]

        return object_contours

    def draw_bboxes(self, image, contours, color=(0,255,0)):
        """Draw bounding boxes around object contours

        Args:
            image (np array): source image for drawing
            contours (_type_): object contours 
            color (tuple, optional): color of bounding boxes. Defaults to (0,255,0).

        Returns:
            _type_: image with object bounding boxes
        """
        dst = image.copy()
        # Loop over the contours
        for c in contours:
            # Draw the bounding box around the contour
            x, y, w, h = cv2.boundingRect(c)
            
            cv2.rectangle(dst, (x, y), (x + w, y + h), color, 2)

        return dst
    
    def export_result_image(self, out_img, output_img_path="./results/three_ships_boxed.tiff"):
        """Save result image into specific path

        Args:
            out_img (_type_): output image array
            output_img_path (str, optional): output image path. Defaults to "./results/three_ships_boxed.tiff".
        """
        # create base folder if not existed
        base_folder = os.path.dirname(output_img_path)
        self.create_folder(base_folder)
        # write output images
        cv2.imwrite(output_img_path, out_img)

    def run(self):
        # read image
        self.src_image = cv2.imread(self.image_path)

        # convert to grayscale
        self.src_gray = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection (Canny)
        self.src_edges = cv2.Canny(self.src_gray, self.opt_args.canny_thres[0], self.opt_args.canny_thres[1])

        # detect lines using HoughTransform
        lines = self.line_detection(self.src_edges, threshold=self.opt_args.line_detection_thres)

        # rotate image to be level
        self.rotated_image, rotated_angle_degree = self.rotate_image_to_be_level(lines, 
                                                                                 self.src_image, 
                                                                                 thres_dist=self.opt_args.hough_dist_thres)

        # crop down image to remove black pixels
        self.image_rotated_cropped = self.crop_image_around_center( self.rotated_image,
                                                                    *self.get_largest_rotated_rect(
                                                                            self.src_image.shape[1],
                                                                            self.src_image.shape[0],
                                                                            math.radians(rotated_angle_degree)
                                                                        )
                                                                    )
                
        # detect contours
        ship_contours = self.extract_contours(  self.image_rotated_cropped,
                                                bilateral_kernel_size=self.opt_args.bilateral_kernel_size,
                                                bilateral_sigma_color=self.opt_args.bilateral_sigma_color,
                                                bilateral_sigma_space=self.opt_args.bilateral_sigma_space,
                                                morpho_kernel_size = self.opt_args.morphology_kernel_size,
                                                opening_iter= self.opt_args.opening_iter,
                                                dilation_iter=self.opt_args.dilation_iter,
                                                upper_hsv = self.opt_args.upper_hsv,
                                                lower_hsv = self.opt_args.lower_hsv,
                                                thres_contour_area = self.opt_args.thres_contour_area
                                                )
        
        # draw bboxes
        result_img = self.draw_bboxes(self.image_rotated_cropped,
                                      contours=ship_contours,
                                      color=self.opt_args.color)
        
        # save images
        self.export_result_image(result_img, output_img_path=self.output_image_path)

        cv2.imshow("Result Image", result_img)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--canny-thres', nargs=2,
                        type=int,
                        default=[75, 150],
                        help='Lower and Upper thresholds for Canny Edge detection')
    
    parser.add_argument('--line-detection-thres',
                        type=int,
                        default=200,
                        help='Threshold to detect line used for HoughTransform')
    
    parser.add_argument('--hough-dist-thres',
                        type=int,
                        default=500,
                        help='Hough radius distance threshold')
    
    parser.add_argument('--bilateral-kernel-size',
                        type=int,
                        default=9,
                        help='Bilateral kernel size for filtering image')
    
    parser.add_argument('--bilateral-sigma-color',
                        type=int,
                        default=75,
                        help='Bilateral sigma color for filtering image')
    
    parser.add_argument('--bilateral-sigma-space',
                        type=int,
                        default=75,
                        help='Bilateral sigma space for filtering image')
    
    parser.add_argument('--morphology-kernel-size', nargs=2,
                        type=int,
                        default=(3, 3),
                        help='Bilateral sigma space for filtering image')
    
    parser.add_argument('--opening-iter',
                        type=int,
                        default=3,
                        help='Opening iteration')
    
    parser.add_argument('--dilation-iter',
                        type=int,
                        default=3,
                        help='Dilation iteration')
    
    parser.add_argument('--upper-hsv', nargs=3,
                        type=int,
                        default=[120, 116, 192],
                        help='Upper thresholds for HSV Color Space')
    
    parser.add_argument('--lower-hsv', nargs=3,
                        type=int,
                        default=[106, 44, 142],
                        help='Lower thresholds for HSV Color Space')
    
    parser.add_argument('--thres-contour-area',
                        type=int,
                        default=750,
                        help='Contour area threshold')
    
    parser.add_argument('--color', nargs=3,
                        type=int,
                        default=(0, 255, 0),
                        help='Color for drawing boxes. Default: green')
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    
    # NOTE. Run `hsv_calibration.py`script to calibrate lower and upper hsv values for extracting desired mask

    # get argparse
    opt = parse_opt()

    # declare ShipDetection object    
    ship_detection = ShipDetection(opt_args=opt,
                                   image_path="./three_ships_horizon.JPG",
                                   output_image_path="./results/three_ships_boxed.tiff")
    # run solution
    ship_detection.run()