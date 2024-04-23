import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import skimage
import matplotlib


def image_displaying(source, canny, finale_p, finale, parametric, mn, mx, nmbr):
    fig,  axs = plt.subplot_mosaic([['src', 'can'], ['fin', 'finp'], ['p', 'info']], layout='tight', figsize=(9.2, 7.8))
    axs['src'].imshow(cv.cvtColor(source, cv.COLOR_BGR2RGB))
    axs['can'].imshow(canny, cmap=matplotlib.cm.gray)
    axs['fin'].imshow(cv.cvtColor(finale, cv.COLOR_BGR2RGB))
    axs['finp'].imshow(cv.cvtColor(finale_p, cv.COLOR_BGR2RGB))
    axs['p'].imshow(cv.resize(parametric.astype(np.float32) / np.max(parametric),
                               (parametric.shape[1], 300)), cmap=matplotlib.cm.gray)
    axs['src'].axis('off')
    axs['can'].axis('off')
    axs['fin'].axis('off')
    axs['finp'].axis('off')
    axs['p'].axis('off')
    axs['info'].axis('off')
    axs['src'].set_title('Image')
    axs['can'].set_title('Canny edges')
    axs['fin'].set_title('Hough')
    axs['finp'].set_title('Probabilistic Hough')
    axs['p'].set_title('Parameter space')
    axs['info'].text(0.5, 0.5, f'Number of lines: {nmbr}\nShortest line: {mn}\nLongest line: {mx}', size=25,
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)
                               )
                     )
    plt.show()
    fig.savefig('results/lines/Klinom.jpg')
    plt.close()


def line_detection(option, image):
    edged = cv.Canny(image, 450, 470, None, 3)
    match option:
        case 0:
            cv.imshow('Source Image', image)
            cv.waitKey(0)
        case 1:
            result = image.copy()
            result_p = image.copy()
            lines = cv.HoughLines(edged, 1, np.pi / 180, 150)
            lines_p = cv.HoughLinesP(edged, 1, np.pi / 180, 50, None, 30, 4)
            angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            ah, _, _ = skimage.transform.hough_line(edged, theta=angles)
            mx = 0
            mn = 10000
            for i in range(len(lines_p)):
                line = lines_p[i][0]
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                cv.line(result_p, pt1, pt2, (0, 255, 255), 5, cv.LINE_AA)
                cv.circle(result_p, pt1, 5, (0, 0, 255), -1)
                cv.circle(result_p, pt2, 5, (0, 0, 255), -1)
                norm = np.linalg.norm(np.array(pt2)-np.array(pt1))
                if norm < mn:
                    mn = round(norm, 2)
                if norm > mx:
                    mx = round(norm, 2)
            for j in range(len(lines)):
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
    result = image.copy()
    edged = cv.Canny(image, 400, 450, None, 3)
    match option:
        case 0:
            cv.imshow('Source Image', image)
            cv.waitKey(0)
        case 1:
            hough_radius = np.arange(175, 185)
            hough_res = skimage.transform.hough_circle(edged, hough_radius)
            ha, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, hough_radius, total_num_peaks=5)
            for center_y, center_x, radius in zip(cy, cx, radii):
                cv.circle(result, (center_x, center_y), int(radius), (128, 0, 128), 2, cv.LINE_AA)
            image_displaying(image, edged, result, result, ha, cx, cy, cx)
        case _:
            print('Wrong option! Enter the right number')


src = cv.imread("images/lines_detection/Klinom Krasnym Bej Belych.jpeg")
if src is None:
    print('Opps! Error opening image =(')
    sys.exit("Could not read the image.")

line_detection(1, src)



