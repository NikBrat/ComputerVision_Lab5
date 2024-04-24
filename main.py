import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import skimage
import matplotlib


def image_displaying(source, canny, finale, finale_r, sp, mn, mx):
    # creating figure with mosaic layout
    fig, axs = plt.subplot_mosaic([['src', 'can'], ['fin', 'finr'], ['info', 'infor']], layout='tight',
                                  figsize=(9.2, 7.8))
    # plotting images
    axs['src'].imshow(cv.cvtColor(source, cv.COLOR_BGR2RGB))
    axs['can'].imshow(canny, cmap=matplotlib.cm.gray)
    axs['fin'].imshow(cv.cvtColor(finale, cv.COLOR_BGR2RGB))
    axs['finr'].imshow(cv.cvtColor(finale_r, cv.COLOR_BGR2RGB))
    # turning off axes
    axs['src'].axis('off')
    axs['can'].axis('off')
    axs['fin'].axis('off')
    axs['finr'].axis('off')
    axs['info'].axis('off')
    axs['infor'].axis('off')
    axs['src'].set_title('Image')
    axs['can'].set_title('Canny edges')
    axs['fin'].set_title('Hough (Specified radius)')
    axs['finr'].set_title(' Hough (Range)')
    # plotting the specified radius size
    axs['info'].text(0.5, 0.5, f'Radius length:{sp}', size=25,
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)
                               )
                     )
    # plotting minimum and maximum radius sizes
    axs['infor'].text(0.5, 0.5, f'Min radius: {mn}\nMax line: {mx}', size=25,
                      ha='center',
                      va='center',
                      bbox=dict(boxstyle="square",
                                ec=(1., 0.5, 0.5),
                                fc=(1., 0.8, 0.8)
                                )
                      )
    # displaying and saving the plot
    plt.show()
    fig.savefig('results/circles/olympic.jpg')
    plt.close()


def line_detection(option, image):
    # Canny edges
    edged = cv.Canny(image, 250, 270, None, 3)
    match option:
        case 0:
            # output of the original image
            cv.imshow('Source Image', image)
            cv.waitKey(0)
        case 1:
            # Hough transform
            # creating image copies
            result = image.copy()
            result_p = image.copy()
            # computing lines via classic Hough transform
            lines = cv.HoughLines(edged, 1, np.pi / 180, 150)
            # computing lines via probabilistic Hough transform
            lines_p = cv.HoughLinesP(edged, 1, np.pi / 180, 50, None, 30, 4)
            #
            angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            # Accumulator computing
            ah, _, _ = skimage.transform.hough_line(edged, theta=angles)
            mx = 0
            mn = 10000
            for i in range(len(lines_p)):
                # overlaying lines found by using probabilistic transform and their edges
                line = lines_p[i][0]
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                cv.line(result_p, pt1, pt2, (0, 255, 255), 5, cv.LINE_AA)
                cv.circle(result_p, pt1, 5, (0, 0, 255), -1)
                cv.circle(result_p, pt2, 5, (0, 0, 255), -1)
                # computing the shortest and longest lines
                norm = np.linalg.norm(np.array(pt2) - np.array(pt1))
                if norm < mn:
                    mn = round(norm, 2)
                if norm > mx:
                    mx = round(norm, 2)
            for j in range(len(lines)):
                # overlaying lines found by using classic transform and their edges
                rho = lines[j][0][0]
                theta = lines[j][0][1]
                a, b = math.cos(theta), math.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (np.int32(x0 + 1000 * (-b)), np.int32(y0 + 1000 * a))
                pt2 = (np.int32(x0 - 1000 * (-b)), np.int32(y0 - 1000 * a))
                cv.line(result, pt1, pt2, (0, 255, 255), 5, cv.LINE_AA)
                cv.circle(result, pt1, 5, (255, 0, 0), -1)
                cv.circle(result, pt2, 5, (255, 0, 0), -1)
            # cv.imwrite('results/lines/Klinom_p.jpg', result_p)
            # transferring recieved images and data to function of creating a comparative image
            image_displaying(image, edged, result_p, result, ah, mn, mx, len(lines_p))
        case 2:
            canny = skimage.feature.canny(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 3, 15, 75)
            lines_p = skimage.transform.probabilistic_hough_line(canny, 50, 10, 4)
            if lines_p is not None:
                for line in lines_p:
                    pt1, pt2 = line
                    cv.line(image, pt1, pt2, (255, 255, 0), 5, cv.LINE_AA)
            image_displaying(image, 'Probabilistic Hough SKIMAGE')
            plt.figure()
            plt.imshow(canny, cmap=matplotlib.cm.gray)
            plt.waitforbuttonpress(0)
        case _:
            print('Wrong option! Enter the right number')


def circle_detection(option, image):
    # creating image copies
    result = image.copy()
    result_r = image.copy()
    edged = image
    match option:
        case 0:
            # output of the original image
            cv.imshow('Source Image', image)
            cv.waitKey(0)
        case 1:
            # Hough transform
            # Specified radius
            hough_radius = np.arange(180, 181)
            # Getting Hough accumulator
            hough_res = skimage.transform.hough_circle(edged, hough_radius)
            # Getting coordinates for centers of circles
            ha, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, hough_radius, total_num_peaks=1)
            for center_y, center_x, radius in zip(cy, cx, radii):
                # overlaying circles of specified radius
                cv.circle(result, (center_x, center_y), int(radius), (255, 255, 0), 10, cv.LINE_AA)
            # Radius range
            hough_radius_r = np.arange(175, 239)
            # Getting Hough accumulator
            hough_res_r = skimage.transform.hough_circle(edged, hough_radius_r)
            # Getting coordinates for centers of circles
            ha, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res_r, hough_radius_r, total_num_peaks=4)
            for center_y, center_x, radius in zip(cy, cx, radii):
                # overlaying circles
                cv.circle(result_r, (center_x, center_y), int(radius), (255, 255, 0), 10, cv.LINE_AA)
            # transferring recieved images and data to function of creating a comparative image
            image_displaying(image, edged, result, result_r, hough_radius[0], hough_radius_r[0], hough_radius_r[-1])
            # cv.imwrite('results/circles/Mercedes_SP.jpg', result)
            # cv.imwrite('results/circles/Mercedes_R.jpg', result_r)
        case _:
            print('Wrong option! Enter the right number')


src = cv.imread("images/circles_detection/olympic.jpeg", 0)
if src is None:
    print('Opps! Error opening image =(')
    sys.exit("Could not read the image.")

circle_detection(1, src)
