from cv2 import cv2 as cv
import numpy as np
import imageparser



def solveImage(frame, dic = None):

    binaryImage = imageparser.binarizeImage(frame)
    corners = imageparser.findCorners(binaryImage)
    warped, dimension = imageparser.getWarped(corners, binaryImage)
    sudokuGrid, original = imageparser.readSudoku(warped, 450) #tesseract works best with 30 pixel high text
    solved, original = imageparser.solveSudoku(sudokuGrid, original, dic)

    if solved is not False:
        sudokuGrid = imageparser.solutionToMatrix(solved, sudokuGrid)
        return imageparser.writeSudoku(original, sudokuGrid, dimension, corners, frame)
    else:
        if dic == None:
            print("It was not possible to solve the sudoku")
        return frame


def solveVideo(path):
    #Function to load the video
    cap = cv.VideoCapture(path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    #FourCC is a 4-byte code used to specify the video codec.
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output2.avi',fourcc, fps, (width,  height))

    #Declaration of the dictionary of solved sudokus
    dic = {}

    while(cap.isOpened()):
        #ret is false if no frame is captured
        ret, frame = cap.read()
        if ret:

            frame = solveImage(frame, dic)

            # write the flipped frame
            out.write(frame)

        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()


path = input("Insert the path to the file -> ")
#Esempio di path 'C:\\Users\\giuly\\Desktop\\sudoku\\video2.avi'

if path.endswith(".avi") or path.endswith(".mp4"):
    solveVideo(path)
elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
    frame = cv.imread(path)
    solved = solveImage(frame)
    cv.imwrite("solved.png", solved)