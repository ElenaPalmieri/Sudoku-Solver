from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import solver
import sudokudict


#Converts the image to grayscale, binarizes it and improves the image quality
def binarizeImage(frame):
    #Conversion of the image to Grayscale
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    imgBlur = cv.GaussianBlur(img, (9, 9), 0)

    #TEST
    # plt.imshow(imgBlur, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    imgBinarized = cv.adaptiveThreshold(imgBlur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    #TEST
    # plt.imshow(imgBinarized, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    imgInverted = cv.bitwise_not(imgBinarized)

    kernel = np.ones((3,3),np.uint8)

    result =  cv.morphologyEx(imgInverted, cv.MORPH_OPEN, kernel)

    #TEST
    # plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    return result


#Finds the corners of the contour with the largest area, that is the main contour of the sudoku
def findCorners(img):
    (_, contours, _) = cv.findContours(img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    #TEST
    # backtorgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # cv.drawContours(backtorgb, contours, -1, (0,255,0), 3)
    # plt.imshow(backtorgb, vmin=0, vmax=255)
    # plt.show()


    largest_area = 0
    largest_contour_index = 0

    #Cycles on every contour and finds the contour with the largest area
    for i in range(0,len(contours)):
        area = cv.contourArea(contours[i])
        if (area>largest_area):
            largest_area = area
            largest_contour_index = i


    minsum=5000
    mindiff=5000
    maxsum=-5000
    maxdiff=-5000

    #There are 4 corners in the largest contour, each with 2 coordinates
    corners = np.zeros((4,2), dtype='float32')

    for point in contours[largest_contour_index]:
        x=float(point[0][0])
        y=float(point[0][1])

        #Top left corner
        if(x+y<minsum):
            minsum=x+y
            corners[0]=point
        #Bottom left corner
        if(x-y<mindiff):
            mindiff=x-y
            corners[1]=point
        #Bottom right corner
        if(x+y>maxsum):
            maxsum=x+y
            corners[2]=point
        #Top right corner
        if(x-y>maxdiff):
            maxdiff=x-y
            corners[3]=point

    #TEST: draw the four corners on the image
    # cv.circle(backtorgb, (corners[0][0],corners[0][1]), 5, (0,0,255), thickness=5)
    # cv.circle(backtorgb, (corners[1][0],corners[1][1]), 5, (0,0,255), thickness=5)
    # cv.circle(backtorgb, (corners[2][0],corners[2][1]), 5, (0,0,255), thickness=5)
    # cv.circle(backtorgb, (corners[3][0],corners[3][1]), 5, (0,0,255), thickness=5)
    # plt.imshow(backtorgb, vmin=0, vmax=255)
    # plt.show()

    return corners

#Squares the grid of the sudoku
def getWarped(corners, img):

    rect = cv.minAreaRect(corners)
    center, size, theta = rect

    # Angle correction
    if theta < -45:
        theta += 90

    rect = (center, size, theta)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    dst_pts = np.float32([[0,0],[0,height],[width,height],[width,0]])

    # the perspective transformation matrix
    m = cv.getPerspectiveTransform(corners, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv.warpPerspective(img, m, (width, height))
    result = cv.resize(warped, (max(width, height), max(width, height)), interpolation=cv.INTER_CUBIC)

    #TEST
    # plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    return result, max(width, height), dst_pts


#Creates a matrix containing the numbers contained in the unsolved sudoku image
def readSudoku(img, dimension):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    cellDim = int(dimension/9)
    margin = int(cellDim/7)

    sudokugrid = np.zeros((9,9), dtype=int)
    original = np.zeros((9,9), dtype=int)

    #Tesseract reads black text on white background
    sudokuW = cv.bitwise_not(img)
    sudokuWhite =  cv.resize(sudokuW, (dimension,dimension)) 

    for i in range(9):
        row = i*cellDim
        for j in range(9):
            col = j*cellDim
            #Takes one of the 81 cells and adds a white padding of dimension "margin" to improve Tesseract's performance
            cell = cv.copyMakeBorder(sudokuWhite[row+margin:row+(cellDim-margin), col+margin:col+(cellDim-margin)], margin*2, margin*2, margin*2, margin*2, cv.BORDER_CONSTANT, value=(255, 255, 255))

            #TEST
            #cv.imwrite((str(i) + str(j) + '.png'), cell)


            txt = pytesseract.image_to_string(cell, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789').strip()
            if txt != '':
                sudokugrid[i][j] = int(txt[0])
                original[i][j] = int(txt[0])

    print("Read sudoku:")
    print(sudokugrid)
    return sudokugrid, original


#Solves the sudoku itself
def solveSudoku(sudokugrid, original, dic = None):
    zeros = 0
    
    #The solver uses a string instead of a matrix
    sudokustring = ""
    for i in range(9):
        for j in range(9):
            if sudokugrid[i][j] == 0:
                zeros+=1
            sudokustring += str(sudokugrid[i][j])

    #To be able to solve univocally a sudoku you need at least 16 numbers on the grid
    if zeros > 65:
        solved = False
    else:
    #If the image is a frame of the video searches the sudoku in the dictionary
        if dic:
            (key, solved) = sudokudict.search(dic, sudokustring)
             #print("search: " + str(solved))
        else:
            solved = False

        #If there is no solution of the sudoku it tries to solve the sudoku
        if solved is False:
            solved = solver.solve(sudokustring)
            #If the sudoku is solved correctly the solution is added to the dictionary
            if solved is not False and dic is not None:
                sudokudict.add(dic, sudokustring, solved)
        #Takes the original reading of the unsolved grid from the dictionary, otherwise the parameter will remain equal to the current reading sudokuGrid
        else:
            original = sudokudict.getgrid(key)

    return solved, original


#The solved sudoku comes in a dictionary with coordinates [A-I][1-9], the loop converts it to a matrix
def solutionToMatrix(solved, sudokuGrid):
    dictionary = {'A': 0, 'B':1, 'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8}
    for i in ['A', 'B', 'C','D','E','F','G','H','I']:
        for j in range(1,10):
            sudokuGrid[dictionary[i]][j-1] = int(solved[i+str(j)])

    print("\n\nSolved sudoku:")
    print(sudokuGrid)

    return sudokuGrid


#Writes the solution on the original image
def writeSudoku(original, sudokuGrid, dimension, dst, corners, frame):

    #sudokuWhite is a layer on which the numbers composing the solution will be written and that will be overlapped to the original image frame
    sudokuWhite = np.zeros((dimension, dimension,3))

    #The values of "col" and "row" are thought to be as centered in the cell as possible
    col=int((7*(dimension/9))/10)
    for i in range(9):
        row=int((3*(dimension/9))/10)
        for j in range(9):
            #Only if the original reading of the grid the cell was empty, the number is written on the image
            if original[i][j]==0 :
                fontdim =  dimension/500
                cv.putText(sudokuWhite, str(sudokuGrid[i][j]), (row, col), cv.FONT_HERSHEY_DUPLEX , fontdim, color = (0, 0, 255) , thickness=2)
            row = row + int(dimension/9)
        col = col + int(dimension/9)

    #TEST
    # plt.imshow(sudokuWhite, vmin=0, vmax=255)
    # plt.show()

    #Undo the warp perspective to match the original image
    m = cv.getPerspectiveTransform(dst,corners)
    solutionLayer = cv.warpPerspective(sudokuWhite, m, (frame.shape[1],frame.shape[0]))

    #TEST
    # plt.imshow(solutionLayer, vmin=0, vmax=255)
    # plt.show()

    #Takes all the pixels in the solutionLayer that have a color value higher than 0 in the third channel
    cnd = solutionLayer[:, :, 2] > 0

    #Substitutes the pixels that are part of the solution with the ones that are in the solutionLayer matrix
    frame[cnd] = solutionLayer[cnd]

    #TEST
    # plt.imshow(frame, vmin=0, vmax=255)
    # plt.show()

    return frame
