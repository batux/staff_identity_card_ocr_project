##################################################
#                                                #
# sahibinden.com Staff Identity Card OCR Project #
# @batux                                         #
#                                                #
##################################################

from os import *
from os.path import *
from PIL import Image
from tesserocr import PyTessBaseAPI

import os
import sys
import cv2
import numpy


def main(args):

    print args

    image_folder_path = "./images/"

    image_names = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "6.jpg", "7.jpg", "8.jpg", "11.jpg", "12.jpg", "13.jpg", "14.jpg"]

    for image_name in image_names:

        #### Open Image ####

        image = open_image(image_folder_path + image_name)
        processable_image = convert_to_processable_format(image)

        #### Open Image ####


        #### Resize Image ####

        resized_image = resize_image(processable_image, 0.15)

        #### Resize Image ####


        #### Gamma Correction ####

        adjusted_image = gamma_correction(resized_image)

        #### Gamma Correction ####


        #### Gray-scale Image ####

        gray_scale_image = convert_to_grayscale_image(adjusted_image)
        negative_image = prepare_negative_image(gray_scale_image)

        #### Gray-scale Image ####


        ###### Edge Detection #####

        cnts, approximated = edge_detection(negative_image)

        ###### Edge Detection #####


        ###### Sort Contours #####

        sorted_cnts, boundingBoxes = sort_contours(approximated)

        ###### Sort Contours #####


        ##### Find OCR Part #####

        last_cnt = find_ocr_part_in_image(sorted_cnts)

        ##### Find OCR Part #####


        #### Perspective Transformation ####

        warped_image = perform_perspective_transformation(last_cnt, processable_image.copy(), resized_image.copy())
        save_image("./images/warped_image.png", warped_image)
        #### Perspective Transformation ####


        #### Find Textblocks ####

        tresh_warped_gray_scale_img, text_blocks = find_text_blocks(warped_image)

        #### Find Textblocks ####


        #### Draw Textblocks ####

        draw_text_blocks(warped_image.copy(), text_blocks, False)

        #### Draw Textblocks ####


        #### OCR Part ####

        run_ocr_engine_for_single_image(tresh_warped_gray_scale_img, text_blocks)

        #### OCR Part ####




def draw_rectangle(img, top_x, top_y, height, width, color, thickness):
    cv2.rectangle(img,(top_x,top_y),(top_x + width, top_y + height), color, thickness)


def perform_adaptive_threshold(gray_scaled_img, x=55, y=10):

    binarized_img = cv2.adaptiveThreshold(gray_scaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, x, y)
    return binarized_img


def open_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGBA")
    return image


def save_image(filename, image):
    cv2.imwrite(filename, image)
    return filename


def convert_to_processable_format(image):
    return cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)


def convert_to_grayscale_image(image):
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_scale_image


def resize_image(image, scale):
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)), cv2.INTER_AREA)
    return resized_image


def prepare_negative_image(image):
    negative_image = 255 - image
    return negative_image


def perform_gaussian_filter(image, kernel=(5,5), constant=0):
    g_image = cv2.GaussianBlur(image, kernel, constant)
    return g_image


def perform_canny_edge_detection(image, min=100, max=200):
    edged_img = cv2.Canny(image, min, max)
    return edged_img


def find_canny_contours(image):

    img_height, img_width = image.shape[:2]

    result = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(result) == 3:
        _, contours, hierarchy = result
    elif len(result) == 2:
        contours, hierarchy = result

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    approximated = []
    min_image_area = (img_height * img_width) * 0.01
    max_image_area = (img_height * img_width) * 0.25

    for contour in cnts:

        contour_area = cv2.contourArea(contour)

        if contour_area > max_image_area:
            continue

        if contour_area > min_image_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            edge_count = len(approx)
            if 3 < edge_count < 13:
                approximated.append(approx)

    approximated = sorted(approximated, key=cv2.contourArea, reverse=True)[:3]

    return cnts, approximated


def sort_contours(contours, i=1, reverse=False):
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)


def prepare_approximated_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = numpy.intp(box)
    return box


def rectangles_contains(rect_1, rect_2):
    return (rect_1[0] > rect_2[0]) and (rect_1[1] > rect_2[1]) and (rect_1[0]+rect_1[2] < rect_2[0]+rect_2[2]) and (rect_1[1]+rect_1[3] < rect_2[1]+rect_2[3])


def rectangles_intersection(rect_1, rect_2):
    x = max(rect_1[0], rect_2[0])
    y = max(rect_1[1], rect_2[1])
    w = min(rect_1[0]+rect_1[2], rect_2[0]+rect_2[2]) - x
    h = min(rect_1[1]+rect_1[3], rect_2[1]+rect_2[3]) - y

    if w < 0 or h < 0:
        return False

    return True


def draw_all_shapes(img, cnts):
    cv2.drawContours(img, cnts, -1, (0,255,0), 3)
    return img


def perform_close_morphology(img, kernel=(5, 5), iteration=1):
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structuring_element, iterations=iteration)
    return closing


def gamma_correction(automatic_resized_image):

    g_sca_image = convert_to_grayscale_image(automatic_resized_image)

    brightness_ratio = calculate_avg_of_brightness(g_sca_image)

    hist = histogram_of_gray_image(g_sca_image)

    i = 0
    total_value = 0
    hist_size = len(hist)
    cdf = create_empty_cdf(hist_size)

    while i < hist_size:
        total_value = total_value + hist[i]
        cdf[i] = total_value
        i = i + 1


    i = 0
    first_index = 0
    while i < len(cdf):
        if cdf[i] != 0:
            first_index = i
            break
        i = i + 1

    max = len(cdf) - 1
    min = first_index

    padding = (max - min) / 2
    median_value = min + padding

    median_brightness_value = (median_value * 1.0) / 255

    gamma_value = calculate_gamma_value(median_brightness_value, brightness_ratio)

    gamma_corrected_image = perform_gamma_correction(automatic_resized_image.copy(), gamma_value)

    #save_image("./images/gamma_corrected{0}.png".format(index), gamma_corrected_image)

    return gamma_corrected_image


def calculate_avg_of_brightness(image):

    h,w = image.shape
    total_brightness = numpy.sum(image[0:h, 0:w])

    avg_brightness = (total_brightness * 1.0) / (h*w)

    brightness_ratio = (avg_brightness * 1.0) / 255

    return brightness_ratio


def histogram_of_gray_image(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return hist


def create_empty_cdf(size):
    cdf = numpy.zeros(size, dtype=int)
    return cdf


def calculate_gamma_value(median_brightness_value, brightness_ratio):
    gamma_value = numpy.log10(median_brightness_value) / numpy.log10(brightness_ratio)
    return gamma_value


def perform_gamma_correction(image, gamma=1.0):
    table = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def perspective_transformation(image, pts, maxWidth=1000, maxHeight=1336):

    rect = order_points(pts)

    dst = numpy.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def order_points(pts):

    rect = numpy.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[numpy.argmin(s)]
    rect[2] = pts[numpy.argmax(s)]

    diff = numpy.diff(pts, axis=1)
    rect[1] = pts[numpy.argmin(diff)]
    rect[3] = pts[numpy.argmax(diff)]

    return rect


def find_text_blocks_in_image(image):

    img_height, img_width = image.shape[:2]

    tresh_of_width = img_width * 0.6
    tresh_of_height = img_height * 0.05

    p = int(image.shape[1] * 0.05)
    image[:, 0:p] = 0
    image[:, image.shape[1] - p:] = 0

    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    cnts, boundingBoxes = sort_contours(cnts)

    text_blocks = []

    biggest_y = 0
    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)

        if y > biggest_y and (h >= 5 and w >= 20):

            if (tresh_of_width > w or tresh_of_height < h):

                biggest_rect = (x, y, w, h)
                text_blocks.append(biggest_rect)


    return text_blocks


def create_structuring_element(kernel):
    structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    return structure_element


def perform_dilation(image, element, iteration=1):
    dilated_img = cv2.dilate(image, element, iterations=iteration)
    return dilated_img


def run_ocr_engine_for_single_image(dst, borders):

    filename = "./images/tmp_{}.png".format(os.getpid())
    cv2.imwrite(filename, dst)

    image = Image.open(filename)
    image.save(filename, dpi=(300, 300))
    image = Image.open(filename)

    with PyTessBaseAPI(path="./tessdata", lang='tur') as api:
        api.SetImage(image)

        print "-----------------------"
        for border in borders:

            x = border[0]
            y = border[1]
            w = border[2]
            h = border[3]

            api.SetRectangle(x, y, w, h)

            result_item = api.GetUTF8Text()

            result_item = result_item.replace("\n\n\n", "\n")
            result_item = result_item.replace("\n\n", "\n")

            print result_item


def edge_detection(negative_image):

    bloor_image = perform_gaussian_filter(negative_image)
    #save_image("./images/proc_image{0}.png".format(index), bloor_image)

    edged_img = perform_canny_edge_detection(bloor_image.copy(), 10, 40)
    #save_image("./images/canny_edged_image{0}.png".format(index), edged_img)


    edged_img = perform_close_morphology(edged_img, (1,10), 1)
    edged_img = perform_close_morphology(edged_img, (10,1), 1)
    #save_image("./images/canny_closed_edged_image{0}.png".format(index), edged_img)

    cnts, approximated = find_canny_contours(edged_img)

    return cnts, approximated


def find_ocr_part_in_image(sorted_cnts):

    length_of_cnts = len(sorted_cnts)

    last_cnt = sorted_cnts[length_of_cnts - 1]

    if(length_of_cnts > 1):

        second_last_cnt = sorted_cnts[length_of_cnts - 2]

        second_last_box = cv2.boundingRect(second_last_cnt)

        last_box = cv2.boundingRect(last_cnt)

        intersection_rect = rectangles_intersection(last_box, second_last_box)

        if intersection_rect:

            contains_rect = rectangles_contains(last_box, second_last_box)

            if contains_rect:
                last_cnt = second_last_cnt

    return last_cnt


def perform_perspective_transformation(last_cnt, processable_image, resized_image=None):

    last_cnt = prepare_approximated_rectangle(last_cnt)

    # If you want to see all contours, you can open this block. It will prepare an image!

    drawed_image = draw_all_shapes(resized_image.copy(), [last_cnt])

    save_image("./images/drawed_image.png", drawed_image)

    last_cnt = (1.0 / 0.15) * last_cnt

    org_image_height, org_image_width, _ = processable_image.shape

    warped_image = perspective_transformation(processable_image, last_cnt.reshape(4, 2), int(org_image_width*0.2), int(org_image_height*0.2))

    return warped_image


def find_text_blocks(warped_image):

    gray_scale_warped_img = convert_to_grayscale_image(warped_image)

    tresh_warped_gray_scale_img = perform_adaptive_threshold(gray_scale_warped_img)

    tresh_warped_gray_scale_img = cv2.bilateralFilter(tresh_warped_gray_scale_img, 15, 60, 60)

    #save_image("./images/warped_image{0}.png".format(index), tresh_warped_gray_scale_img)


    negative_tresh_gray_scale_img = prepare_negative_image(tresh_warped_gray_scale_img)

    structuring_element = create_structuring_element((30, 1))

    dilated_image = perform_dilation(negative_tresh_gray_scale_img, structuring_element, 1)

    #save_image("./images/dilated_warped_image{0}.png".format(index), dilated_image)

    text_blocks = find_text_blocks_in_image(dilated_image)

    return tresh_warped_gray_scale_img, text_blocks


def draw_text_blocks(warped_image, text_blocks, enable=False):

    if enable:

        for textblock in text_blocks:
            draw_rectangle(warped_image, textblock[0], textblock[1], textblock[3], textblock[2], (0,255,0), 3)

        #save_image("./images/drawed_warped_image{0}.png".format(index), copy_img)

    return warped_image


if __name__ == '__main__':
    main(sys.argv)